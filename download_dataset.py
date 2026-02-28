from roboflow import Roboflow

rf = Roboflow(api_key="h9s3uqIudPtt2YcNWSzM")

project = rf.workspace("meet-zhc9j").project("sack-t7ftj")

version = project.version(2)

dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)
