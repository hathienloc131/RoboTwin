# Converts data/third_view/<task>/demo_clean (collected with task_config
# demo_clean_thirdview.yml, data_type.third_view: true -- see
# envs/_base_task.py:get_obs) into LeRobot datasets whose "head_camera" feature
# is populated from the third-person third_view_rgb camera instead of the real
# head camera (--head-camera-source third_view; see
# script/lerobot/convert_robotwin_to_lerobot.py's module docstring). The output
# schema is otherwise identical to convert_data.sh's ("head_camera"/
# "left_camera"/"right_camera" feature names unchanged), so it's a drop-in
# swap for training/eval: pair with GR00TRoboTwinEndposeAdapter(third_view=True)
# / script/eval_gr00t_endpose_thirdview.py.
#
# Each --task-name/--robot-type is passed explicitly because
# data/third_view/<task>/demo_clean's folder name ("demo_clean") doesn't match
# convert_robotwin_to_lerobot.py's <task>_<embodiment>_<config>_<episodes>
# auto-parse pattern.
#
# Each --instruction mirrors convert_data.sh's for the same task (same
# task-level generic instruction, valid across every episode).

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/third_view/grab_roller/demo_clean \
    --output-path data/converted_third_view/grab_roller \
    --task-name grab_roller \
    --robot-type aloha-agilex \
    --cameras head_camera left_camera right_camera \
    --head-camera-source third_view \
    --instruction "Use both arms to grab the roller on the table."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/third_view/adjust_bottle/demo_clean \
    --output-path data/converted_third_view/adjust_bottle \
    --task-name adjust_bottle \
    --robot-type aloha-agilex \
    --cameras head_camera left_camera right_camera \
    --head-camera-source third_view \
    --instruction "Pick up the bottle from the table head-up with the correct arm."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/third_view/open_laptop/demo_clean \
    --output-path data/converted_third_view/open_laptop \
    --task-name open_laptop \
    --robot-type aloha-agilex \
    --cameras head_camera left_camera right_camera \
    --head-camera-source third_view \
    --instruction "Use the appropriate arm to open the laptop."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/third_view/place_object_basket/demo_clean \
    --output-path data/converted_third_view/place_object_basket \
    --task-name place_object_basket \
    --robot-type aloha-agilex \
    --cameras head_camera left_camera right_camera \
    --head-camera-source third_view \
    --instruction "Use one arm to place the object into the basket, then use the other arm to grab the basket and move it slightly away."
