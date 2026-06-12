import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

temporal = rs.temporal_filter()
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

print(type(depth_frame))
filtered = temporal.process(depth_frame)
print(type(filtered))
print(type(filtered.as_depth_frame()))
pipeline.stop()
