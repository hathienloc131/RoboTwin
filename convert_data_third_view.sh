# Each --instruction below is derived from the task's own generic template
# (description/task_instruction/*.json "full_description" field), which
# already avoids per-episode-randomized specifics (which arm, which color)
# so a single instruction is valid across every episode in the dataset.

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/grab_roller0 \
    --output-path data/converted_front/grab_roller \
    --instruction "Use both arms to grab the roller on the table."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/adjust_bottle \
    --output-path data/converted_front/adjust_bottle \
    --instruction "Pick up the bottle from the table head-up with the correct arm."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/move_stapler_pub \
    --output-path data/converted_front/move_stapler_pub \
    --instruction "Use the appropriate arm to move the stapler onto the colored mat."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/open_laptop \
    --output-path data/converted_front/open_laptop \
    --instruction "Use the appropriate arm to open the laptop."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/place_mouse_pad \
    --output-path data/converted_front/place_mouse_pad \
    --instruction "Pick up the mouse and place it on the colored mat using the appropriate arm."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/place_object_basket \
    --output-path data/converted_front/place_object_basket \
    --instruction "Use one arm to place the object into the basket, then use the other arm to grab the basket and move it slightly away."


