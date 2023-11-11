import fire
import os
import api.util as util
import json
import boto3
import os
import tarfile

def make_tar(directory_path, output_filename):
    """Compresses directory into a tar file."""
    with tarfile.open(output_filename, "w") as tar:
        tar.add(directory_path, arcname=os.path.basename(directory_path))

def upload_directory_to_s3(directory_path, bucket_name, tar_name):
    make_tar(directory_path, tar_name)

    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file(tar_name, bucket_name, tar_name)

    print(f"Uploaded {tar_name} to {bucket_name}")
    print("The link is: ", f"https://genies-data.s3.us-east-2.amazonaws.com/genies-datasets.tar")


def upload_data():
    upload_directory_to_s3(f"distributions", "genies-data", f"genies-datasets.tar")

if __name__ == "__main__":
    fire.Fire(upload_data)