# Robotiq EPick live collision model

`urdf/robotiq_epick_active_tcp_collision.urdf` is a fixed collision URDF rooted
at the configured active TCP.  Its EPick body and suction-cup dimensions are
derived from PickNik Robotics' BSD-3-Clause `epick_description`; the rear
mount allowance is the 44.7 mm excess of the right robot's measured 161.85 mm
flange-to-active-TCP distance over the described 117.3 mm body-plus-cup chain.

Before using this model for motion planning, verify the physical mount length
and cup type, then update the `mount_extension` cylinder length accordingly.
