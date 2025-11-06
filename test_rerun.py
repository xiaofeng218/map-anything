import time
import numpy as np
import rerun as rr

rr.init("rerun_example_embed_web_viewer")

positions = np.vstack([xyz.ravel() for xyz in np.mgrid[3 * [slice(-10, 10, 10j)]]]).T
colors = np.vstack([rgb.ravel() for rgb in np.mgrid[3 * [slice(0, 255, 10j)]]]).astype(np.uint8).T

rr.log("my_points", rr.Points3D(positions, colors=colors, radii=0.5))

grpc_uri = rr.serve_grpc() # Hosts the gRPC server
rr.serve_web_viewer(connect_to=grpc_uri) # Hosts a web viewer

try:
  while True:
    time.sleep(1)
except KeyboardInterrupt:
  print("Ctrl-C received. Exiting.")
