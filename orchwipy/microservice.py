import abc
import dataclasses
import io
import json
import os
import logging
import sys
from pathlib import Path, PurePosixPath
from uuid import uuid4
from typing import Any, Optional, TypedDict, Callable

import boto3
import yaml
from botocore.exceptions import ClientError

from .func import Functions, log, PauseExecution

OrchwipyRequestBucket = os.getenv("OrchwipyRequestBucket")
MAX_RETRY = 4

log = logging.getLogger(__name__)


@dataclasses.dataclass
class FunctionOnLambda(Functions):
    _: dataclasses.KW_ONLY
    current_context: Optional["LambdaContext"] = dataclasses.field(default=None)
    template_props: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    @property
    def is_lambda(self):
        return bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

    def fname_to_lambda(self, fname: str):
        cfn = boto3.client("cloudformation")
        stack = os.environ["StackName"]
        res = cfn.describe_stack_resource(
            StackName=stack, LogicalResourceId=cfn_logical_name(fname)
        )
        return res["StackResourceDetail"]["PhysicalResourceId"]

    @property
    def fname(self):
        return os.environ["OrchwipyName"]

    def _register(
        self, *, handler, name, template_props: dict[str, Any] | None = None, **kwargs
    ):
        if template_props:
            self.template_props[name] = template_props
        return super()._register(handler=handler, name=name, **kwargs)

    @staticmethod
    def new_payload_with_s3(**data):
        payload = json.dumps(data).encode()
        s3client = boto3.client("s3")
        payload_key = str(PurePosixPath("orchwipy-payload", str(uuid4())))
        log.debug(
            "Saving payload to s3 %s",
            payload_key,
            extra=dict(key=payload_key),
        )
        s3client.put_object(Bucket=OrchwipyRequestBucket, Body=payload, Key=payload_key)
        payload = json.dumps({"$payloadS3Key": payload_key}).encode()
        return payload

    def do_call(self, *, dest, data):
        if self.is_lambda:
            fname = self.fname_to_lambda(dest)
            log.debug("Invoking %s", fname)
            client = boto3.client("lambda")
            if fname == os.environ["AWS_LAMBDA_FUNCTION_NAME"]:
                log.warning("Cannot invoke own lambda function")
                return dict()
            payload = json.dumps(data).encode()
            try:
                return client.invoke(
                    FunctionName=fname, InvocationType="Event", Payload=payload
                )
            except ClientError as err:
                if err.response["Error"]["Code"] == "RequestEntityTooLargeException":
                    payload = self.new_payload_with_s3(**data)
                    return client.invoke(
                        FunctionName=fname, InvocationType="Event", Payload=payload
                    )
                else:
                    raise

        log.debug("Not lambda")
        out = super().do_call(dest=dest, data=data)
        json.dumps(out)
        return out

    def lambda_handler(self, event, context: "LambdaContext"):
        if payload_s3_key := event.get("$payloadS3Key"):
            log.debug(
                "Loading payload from s3 %s",
                payload_s3_key,
                extra=dict(key=payload_s3_key),
            )
            bucket = boto3.resource("s3").Bucket(OrchwipyRequestBucket)
            buffer = io.BytesIO()
            bucket.download_fileobj(payload_s3_key, buffer)
            buffer.seek(0)
            event = json.load(buffer)

        self.current_context = context
        super().do_call(dest=self.fname, data=event)
        return dict(statusCode=200)


@dataclasses.dataclass
class DelayedRetryFunction(FunctionOnLambda):
    _: dataclasses.KW_ONLY
    delay_queue_name: str | None = dataclasses.field(init=False)
    create_delay_queue_logical_name: str = dataclasses.field(init=False)
    other_resource_templates: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    current_args: dict | None = dataclasses.field(default=None)
    current_retry: int = dataclasses.field(default=0)

    def __post_init__(self):
        self.delay_queue_name = os.environ["DelayQueueURL"] if self.is_lambda else None
        self.create_delay_queue_logical_name = "DelayFnQueue"

        self.other_resource_templates[self.create_delay_queue_logical_name] = dict(
            Type="AWS::SQS::Queue",
            Properties=dict(MessageRetentionPeriod=5 * 60),
        )

        self.other_resource_templates["DelayedFn"] = dict(
            Type="AWS::Serverless::Function",
            Properties=dict(
                Events=dict(
                    OrchwipyRunDelayed=dict(
                        Type="SQS",
                        Properties=dict(
                            Enabled=True,
                            Queue={
                                "Fn::GetAtt": [
                                    self.create_delay_queue_logical_name,
                                    "Arn",
                                ]
                            },
                        ),
                    ),
                ),
                InlineCode=(
                    Path(__file__).parent / "delayed_runner_lambda.py"
                ).read_text(),
                Handler="index.handler",
                Architectures=["x86_64"],
                Runtime="python3.12",
                Timeout=30,
                LoggingConfig=dict(
                    ApplicationLogLevel="INFO",
                    LogFormat="JSON",
                    LogGroup={"Ref": "LogGroup"},
                ),
                Policies=[dict(LambdaInvokePolicy=dict(FunctionName="*"))],
            ),
        )
        return super().__post_init__()

    def update_fn_params(self, prop_key: str, prop_value: Any):
        if prop_key == "Policies":
            assert isinstance(prop_value, list)
            prop_value.append(
                dict(
                    SQSSendMessagePolicy=dict(
                        QueueName={
                            "Fn::GetAtt": [
                                self.create_delay_queue_logical_name,
                                "QueueName",
                            ]
                        }
                    )
                )
            )

    def _register(self, *, handler, name, template_props=None, **kwargs):
        template_props = {**template_props} if template_props else {}
        templ_env = template_props["Environment"] = template_props.get(
            "Environment", {}
        )
        templ_env_vars = templ_env["Variables"] = templ_env.get("Variables", {})
        templ_env_vars["DelayQueueURL"] = {"Ref": self.create_delay_queue_logical_name}

        return super()._register(
            handler=handler, name=name, template_props={**template_props}, **kwargs
        )

    def save_args(self, event, fname):
        self.current_args = event
        return super().save_args(event, fname)

    @property
    def lambda_name(self):
        return os.getenv("AWS_LAMBDA_FUNCTION_NAME")

    @property
    def delay_run_queue(self):
        return os.getenv("DelayQueueURL")

    def call_wrap(self, fn, **args):
        assert self.delay_run_queue if self.lambda_name else True
        try:
            return super().call_wrap(fn, **args)
        except RetryFunctionAfterDelay as ret:
            if self.current_retry >= MAX_RETRY:
                log.warning(
                    "Too many retries: %d, will not retry again" % self.current_retry,
                    extra=dict(retry=self.current_retry, function=self.running_name),
                )
                raise PauseExecution

            assert self.current_args is not None
            if this_lambda := self.lambda_name:
                call_payload = self.new_payload_with_s3(
                    **{**self.current_args, "$retry": self.current_retry + 1}
                ).decode()

                queue = boto3.client("sqs")
                resp = queue.send_message(
                    QueueUrl=self.delay_run_queue,
                    MessageBody=call_payload,
                    DelaySeconds=ret.delay_sec,
                    MessageAttributes={
                        "TargetFn": dict(StringValue=this_lambda, DataType="String")
                    },
                )
                log.debug(
                    "Pause execution",
                    extra=dict(DelaySeconds=ret.delay_sec, function=self.running_name),
                )
                raise PauseExecution

            else:
                raise

    def prologue(self, *, dest, data):
        self.current_retry = data.get("$retry", 0)
        return {**super().prologue(dest=dest, data=data), "$retry": self.current_retry}

    def special_keys(self):
        yield "$retry"
        yield from super().special_keys()


@dataclasses.dataclass
class MicroFunctions(DelayedRetryFunction):

    @classmethod
    def from_functions(cls, fn: Functions, /):
        try:
            return cls(**dataclasses.asdict(fn))
        except TypeError as exc:
            if (
                exc.args[0] == "first argument must be callable or None"
                and sys.version_info.major == 3
                and sys.version_info.minor < 12
            ):
                raise ValueError(
                    "Sorry, python >= 3.12 required for converting Functions to MicroFunctions. Please directly instantiate MicroFunctions instead of Functions"
                )
            raise


def case_snake_to_camel(x: str):
    import string

    change_cap = True
    for ch in x:
        if not str.isalnum(ch):
            change_cap = True
        else:
            if change_cap:
                if ch in string.ascii_uppercase:
                    yield ch.lower()
                    change_cap = False
                elif ch in string.ascii_lowercase:
                    change_cap = False
                    yield ch.upper()
                else:
                    yield ch
            else:
                yield ch


def cfn_logical_name(name: str):
    return "".join(case_snake_to_camel(name))


def build_cfn_template(
    *,
    output_template: io.TextIOWrapper,
    code_py_relative_to_template: Path | str,
    plant: MicroFunctions,
    layers: list[str],
    policies: list,
    environment: Optional[dict] = None,
    more_template_params: Optional[list[str | tuple[str, str]]] = None,
    max_steps_in_run=50,
    timeout=30,
    runtime="python3.11",
    **more_params: dict[str, Any],
):

    more_template_params = list(more_template_params) if more_template_params else []
    more_template_params.append("OrchwipyRequestBucket")
    more_template_params.append(("OrchwipyMaxStack", max_steps_in_run))

    add_policies = [
        {"LambdaInvokePolicy": {"FunctionName": "*"}},
        {"S3ReadPolicy": {"BucketName": {"Ref": "OrchwipyRequestBucket"}}},
        {
            "Statement": [
                {
                    "Action": ["cloudformation:DescribeStackResource"],
                    "Effect": "Allow",
                    "Resource": "*",
                }
            ],
        },
    ]
    environment = dict(**environment) if environment else {}
    environment["OrchwipyMaxStack"] = dict(Ref="OrchwipyMaxStack")

    plant.update_fn_params(prop_key="Policies", prop_value=add_policies)

    policies = [*policies, *add_policies]
    code_py_relative_to_template = Path(code_py_relative_to_template)

    res: dict = dict()

    out_entry: dict = {"Fn::GetAtt": ["", "Arn"]}

    template_params = {l: dict(Type="String") for l in layers}

    for param in more_template_params:
        if isinstance(param, tuple):
            template_params[param[0]] = dict(Type="String", Default=param[1])
        else:
            template_params[param] = dict(Type="String")

    template = dict(
        Transform="AWS::Serverless-2016-10-31",
        Parameters=template_params,
        Resources=res,
        Outputs={"PipelineEntryLambdaArn": {"Value": out_entry}},
    )

    def resource_props(name: str):
        out = dict(
            CodeUri=str(code_py_relative_to_template.parent),
            Handler=code_py_relative_to_template.with_suffix("").name
            + ".lambda_handler",
            Architectures=["x86_64"],
            Runtime=runtime,
            Timeout=timeout,
            Layers=[dict(Ref=layer) for layer in layers],
            Environment=dict(
                Variables=dict(
                    OrchwipyRequestBucket=dict(Ref="OrchwipyRequestBucket"),
                    StackName=dict(Ref="AWS::StackName"),
                    **(environment or {}),
                ),
            ),
            Policies=policies,
            **more_params,
        )
        if template_props := plant.template_props.get(name):
            for k, v in template_props.items():
                if k in out:
                    current = out[k]
                    if isinstance(current, list) and isinstance(v, list):
                        for vv in v:
                            if isinstance(vv, dict):
                                if ref := vv.get("Ref"):
                                    template_params[ref] = dict(Type="String")
                        current.extend(v)
                    elif isinstance(current, str) and isinstance(v, str):
                        out[k] = v
                    elif isinstance(current, dict) and isinstance(v, dict):
                        for vk, vv in v.items():
                            if vk in current:
                                cc = current[vk]
                                if isinstance(cc, dict) and isinstance(vv, dict):
                                    cc.update(vv)
                                else:
                                    raise ValueError(vk)
                            else:
                                current[vk] = vv
                    else:
                        raise ValueError(k)
        return out

    def resource(name: str):
        return dict(Type="AWS::Serverless::Function", Properties=resource_props(name))

    entry_lambda = ""

    for name in plant.functions.keys():
        log_name = cfn_logical_name(name=name)
        if name == plant.beginner:
            entry_lambda = log_name

        assert log_name not in res
        x = res[log_name] = resource(name)
        x["Properties"]["Environment"]["Variables"]["OrchwipyName"] = name

    if conflict_res_names := set(res.keys()).intersection(
        plant.other_resource_templates.keys()
    ):
        raise ValueError(conflict_res_names)

    res.update(plant.other_resource_templates)

    for res_name, res_ in res.items():
        for prop_name, prop in res_["Properties"].items():
            if isinstance(prop, str):
                try:
                    px = prop.format(code_path=code_py_relative_to_template)
                except IndexError:
                    continue
                if px != prop:
                    print("Replacing %s > %s with %s" % (res_name, prop_name, px))
                    res_["Properties"][prop_name] = px

    out_entry["Fn::GetAtt"][0] = entry_lambda

    def str_presenter(dumper, data):
        if "\n" in data:
            block = "\n".join([line.rstrip() for line in data.splitlines()])
            if data.endswith("\n"):
                block += "\n"
            return dumper.represent_scalar("tag:yaml.org,2002:str", block, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

    yaml.safe_dump(
        data=template,
        stream=output_template,
        sort_keys=False,
    )


class RetryFunctionAfterDelay(BaseException):
    def __init__(self, delay_sec: float, *args):
        self.delay_sec = delay_sec
        super().__init__(*args)


class LambdaContext(abc.ABC):
    function_name: str
    aws_request_id: str
    log_group_name: str
    log_stream_name: str

    @abc.abstractmethod
    def get_remaining_time_in_millis(self) -> float:
        pass
