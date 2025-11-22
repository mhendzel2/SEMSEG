
import neuroglancer
import time
import sys

# simple mock if not installed, but we want to fail if it's not real
try:
    import neuroglancer
except ImportError:
    print("neuroglancer not installed")
    sys.exit(1)

def main():
    neuroglancer.set_server_bind_address(bind_address='0.0.0.0', bind_port=9999)
    viewer = neuroglancer.Viewer()
    print(f"Viewer running at {str(viewer)}")

    dataset_id = "jrc_hela-2"
    # Construct the source URL
    # Neuroglancer typically expects 'zarr://s3://bucket/path/to/zarr'
    # OpenOrganelle bucket is 'open-organelle'
    source_url = f"zarr://s3://open-organelle/{dataset_id}/{dataset_id}.zarr"

    print(f"Adding layer from: {source_url}")

    with viewer.txn() as s:
        s.layers['image'] = neuroglancer.ImageLayer(source=source_url)

    print("Layer added. Keeping alive for 5 seconds...")
    time.sleep(5)
    print("Done.")

if __name__ == "__main__":
    main()
