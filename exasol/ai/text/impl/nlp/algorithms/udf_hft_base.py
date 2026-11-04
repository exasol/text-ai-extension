from dataclasses import dataclass

DEVICE_CPU = -1


@dataclass
class UDFHftBase:
    """
    A base class for an algorithm utilising the Huggingface transformers library.

    Parameters:
        model_name:
            The model name as it's seen on the Haggingface model card, for example
            'cross-encoder/nli-deberta-base'.
        bucketfs_conn_name:
            Name of the BucketFS connection object with BucketFS connection credentials.
        sub_dir:
            The subdirectory in the BucketFS where the model is uploaded. The
            subdirectory is relative to the root mounted in the language container.
    """

    model_name: str
    bucketfs_conn_name: str
    sub_dir: str

    @property
    def bucketfs_param_names(self) -> set[str]:
        """
        A list of names of the bucketfs parameters. These parameters shall be considered
        "static", and as such not a part of algorithm configuration.
        """
        return {"bucketfs_conn_name", "sub_dir"}
