# Robotiq 2F-85 live collision model

`urdf/robotiq_2f85_active_tcp_collision.urdf` is a conservative fixed-joint
collision envelope rooted at the configured active TCP, not a visual CAD mesh.
The body and two finger envelopes include the entire unobserved finger sweep;
the model is therefore appropriate for RGB-D robot-pixel removal and obstacle
modelling when no reliable gripper joint angle is published.

The link naming, 2F-85 geometry convention, and collision-model approach were
cross-checked against the BSD-3-Clause `robotiq_description` project from
PickNik Robotics.  The local URDF is independently authored with primitive
collision geometry so the live pipeline needs neither ROS nor xacro.
