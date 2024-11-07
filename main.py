import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Define start and end points for the arrows
start_1 = (0, 0)
end_1 = (0, 100)
start_2 = (100, 0)
end_2 = (100, 100)

headVec = np.array(end_1) - np.array(start_1)
tailVec = np.array(start_2) - np.array(end_2)

# Get the min and max coordinates to set limits dynamically
x_min = min(start_1[0], end_1[0], start_2[0], end_2[0])
x_max = max(start_1[0], end_1[0], start_2[0], end_2[0])
y_min = min(start_1[1], end_1[1], start_2[1], end_2[1])
y_max = max(start_1[1], end_1[1], start_2[1], end_2[1])

# Add some margin to make the arrows more visible within the canvas
margin = 100
x_min -= margin
x_max += margin
y_min -= margin
y_max += margin

# Calculate arrowhead size based on arrow length
def calculate_arrowhead_size(start, end):
    length = np.hypot(end[0] - start[0], end[1] - start[1])
    head_width = 0.05 * length
    head_length = 0.1 * length
    return head_width, head_length

# Set up the canvas
fig, ax = plt.subplots()
#ax.set_xlim(x_min, x_max)
#ax.set_ylim(y_min, y_max)

# Draw the arrows with dynamic head sizes
head_width_1, head_length_1 = calculate_arrowhead_size(start_1, end_1)
ax.arrow(start_1[0], start_1[1], end_1[0] - start_1[0], end_1[1] - start_1[1],head_width=head_width_1, head_length=head_length_1, fc='blue', ec='blue', length_includes_head=True)

head_width_2, head_length_2 = calculate_arrowhead_size(start_2, end_2)
ax.arrow(start_2[0], start_2[1], end_2[0] - start_2[0], end_2[1] - start_2[1],head_width=head_width_2, head_length=head_length_2, fc='red', ec='red', length_includes_head=True)


# Define input for the magnitude and angle
mag = 3
theta = 5

# Initialize the line to animate
line = None
stopProgram=False
curr="head"
lastHeadVec=headVec
lastTailVec=tailVec
lastHeadPoint=end_1
lastTailPoint=start_2

def frame_generator():
    global stopProgram
    i=0
    while stopProgram!=True:  # Define your stopping condition
        yield i
        i += 1

def distanceBetween(startPoint,endPoint):
    return np.linalg.norm(np.array(endPoint)-np.array(startPoint))

def forwardMov(lastVec,start,angleToForward):
    unit_direction = lastVec / np.linalg.norm(lastVec)
    # Step 3: Scale to the new magnitude
    new_vector = mag * unit_direction

    theta_rad = np.radians(angleToForward)  # Convert to radians
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],[np.sin(theta_rad),  np.cos(theta_rad)]])
    rotated_direction = rotation_matrix.dot(new_vector)
    new_end1 = start + rotated_direction

    theta_rad = np.radians(-1*angleToForward)  # Convert to radians
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],[np.sin(theta_rad),  np.cos(theta_rad)]])
    rotated_direction = rotation_matrix.dot(new_vector)
    new_end2 = start + rotated_direction

    return new_end1,new_end2

# Animation function
def animate(frame):
    global line,curr,lastHeadVec,lastTailVec,lastHeadPoint,lastTailPoint,stopProgram
    tempLine=np.array(lastTailPoint) - np.array(lastHeadPoint)
    if(curr=="head"):
        if(distanceBetween(lastHeadPoint,lastTailPoint)<mag):
            stopProgram=True
        else:
            dot_product = np.dot(lastHeadVec, tempLine)

            # Calculate the magnitudes (lengths) of the two vectors
            old_line_magnitude = np.linalg.norm(lastHeadVec)
            new_line_magnitude = np.linalg.norm(tempLine)

            # Calculate the cosine of the angle
            cos_angle = dot_product / (old_line_magnitude * new_line_magnitude)

            # Calculate the angle in radians
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            angleToForward=min(abs(theta),abs(angle_degrees))
            tempNewHeadPoint1,tempNewHeadPoint2=forwardMov(lastHeadVec,lastHeadPoint,angleToForward)

            disTempPt1=distanceBetween(tempNewHeadPoint1,lastTailPoint)
            disTempPt2=distanceBetween(tempNewHeadPoint2,lastTailPoint)

            tempNewHeadPoint=tempNewHeadPoint1
            if(disTempPt2<disTempPt1):
                tempNewHeadPoint=tempNewHeadPoint2

            line=ax.plot([lastHeadPoint[0], tempNewHeadPoint[0]], [lastHeadPoint[1], tempNewHeadPoint[1]], c="red", linewidth=1)[0]
            lastHeadVec=np.array(tempNewHeadPoint)-np.array(lastHeadPoint)
            lastHeadPoint=tempNewHeadPoint
            curr="tail"
    else:
        if(distanceBetween(lastHeadPoint,lastTailPoint)<mag):
            stopProgram=True
        else:
            dot_product = np.dot(lastTailVec, tempLine)

            # Calculate the magnitudes (lengths) of the two vectors
            old_line_magnitude = np.linalg.norm(lastTailVec)
            new_line_magnitude = np.linalg.norm(tempLine)

            # Calculate the cosine of the angle
            cos_angle = dot_product / (old_line_magnitude * new_line_magnitude)

            # Calculate the angle in radians
            angle_radians = np.arccos(cos_angle)
            angle_degrees = np.degrees(angle_radians)
            angleToForward=min(abs(theta),abs(angle_degrees))
            tempNewTailPoint1,tempNewTailPoint2=forwardMov(lastTailVec,lastTailPoint,angleToForward)

            disTempPt1=distanceBetween(tempNewTailPoint1,lastHeadPoint)
            disTempPt2=distanceBetween(tempNewTailPoint2,lastHeadPoint)

            tempNewTailPoint=tempNewTailPoint1
            if(disTempPt2<disTempPt1):
                tempNewTailPoint=tempNewTailPoint2

            line=ax.plot([lastTailPoint[0], tempNewTailPoint[0]], [lastTailPoint[1], tempNewTailPoint[1]], c="red", linewidth=1)[0]
            lastTailVec=np.array(tempNewTailPoint)-np.array(lastTailPoint)
            lastTailPoint=tempNewTailPoint
            curr="head"
    ax.relim()            # Recalculate limits based on the current data
    ax.autoscale_view()   # Apply the new limits
    return line,

# Create the animation
ani = FuncAnimation(fig, animate,frames=frame_generator, interval=5, repeat=False,cache_frame_data=False)

# Display the canvas
plt.show()
