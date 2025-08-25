import logging

log = logging.getLogger(__name__)

import boto3

lambda_client = boto3.client("lambda")


def handler(event, context):
    for record in event.get("Records"):
        if isinstance(record, dict):
            target_fname = (
                record.get("messageAttributes", {})
                .get("TargetFn", {})
                .get("stringValue")
            )
            if isinstance(target_fname, str) and target_fname:
                payload = str(record["body"]).encode()
                log.debug("Invoking %s", target_fname, extra=dict(fname=target_fname))
                lambda_client.invoke(
                    FunctionName=target_fname, InvocationType="Event", Payload=payload
                )
            else:
                log.warning(
                    "NOP",
                    extra=dict(
                        attribs=record["messageAttributes"], recordBody=record["body"]
                    ),
                )
