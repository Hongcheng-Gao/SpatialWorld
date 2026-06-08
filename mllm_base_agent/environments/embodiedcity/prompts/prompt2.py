

def build_prompt(navi_desc):
    TaskDescription = f"""
Please play the role of a drone pilot. The drone is carrying takeout and needs
to move to the location described by the customer.

I will provide the current drone gimbal angle (0 degrees for horizontal view and
90 degrees for top view), plus RGB and depth images. The depth image is black
and white; darker colors indicate closer distances.

Available drone commands:
1. stop
2. moveForth  # Move forward one unit
3. moveUp     # Move up one unit
4. moveDown   # Move down one unit
5. turnLeft   # Rotate 90 degrees to the left
6. turnRight  # Rotate 90 degrees to the right

One unit is 10 meters. To move left, turn left and then move forward. Maintain
a useful top-down or horizontal view of the target building or location.

Navigation instruction: {navi_desc}

Avoid constantly spinning in place. Return one command at a time.
"""
    return TaskDescription