# Each --instruction below is derived from the task's own generic template
# (description/task_instruction/*.json "full_description" field), which
# already avoids per-episode-randomized specifics (which arm, which color)
# so a single instruction is valid across every episode in the dataset.

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/grab_roller_aloha-agilex_clean_50 \
    --output-path data/converted/grab_roller \
    --instruction "Use both arms to grab the roller on the table."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/adjust_bottle_aloha-agilex_clean_50 \
    --output-path data/converted/adjust_bottle \
    --instruction "Pick up the bottle from the table head-up with the correct arm."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/blocks_ranking_rgb_aloha-agilex_clean_50 \
    --output-path data/converted/block_ranking_rgb \
    --instruction "Place the red block, green block, and blue block in order from left to right, placing them in a row."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/move_stapler_pub_aloha-agilex_clean_50 \
    --output-path data/converted/move_stapler_pub \
    --instruction "Use the appropriate arm to move the stapler onto the colored mat."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/open_laptop_aloha-agilex_clean_50 \
    --output-path data/converted/open_laptop \
    --instruction "Use the appropriate arm to open the laptop."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/place_mouse_pad_aloha-agilex_clean_50 \
    --output-path data/converted/place_mouse_pad \
    --instruction "Pick up the mouse and place it on the colored mat using the appropriate arm."

python script/lerobot/convert_robotwin_to_lerobot.py \
    --src-path data/place_object_basket_aloha-agilex_clean_50 \
    --output-path data/converted/place_object_basket \
    --instruction "Use one arm to place the object into the basket, then use the other arm to grab the basket and move it slightly away."


