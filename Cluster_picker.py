from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
if __name__ == "__main__":
    # get gaze data
    point_path = "D:/MASC/shot_processed_dataset/gaze/Allison Ungar Self Tape_0.pkl"
    with open(point_path, "rb") as f:
        data = pkl.load(f)
    out_path = "D:/MASC/JALI_gaze/Animations/heat/annotated_scene/heat_source_video_points_1.json"
    # Create a scatter plot of the data
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1])

    # Define a variable to store the circle patch
    circle = None
    output = []
    speaker = -1

    # Define a function to handle the mouse click event
    def onclick(event):
        # Get the x and y coordinates of the mouse click
        x, y = event.xdata, event.ydata
        # Check if the click is within the axes limits
        if x is not None and y is not None:
            # Define the initial radius of the circle
            radius = 0.1

            # Create a circle patch with the specified centroid and radius
            global circle
            circle = plt.Circle((x, y), radius, color='r', fill=False)

            # Add the circle patch to the plot
            ax.add_patch(circle)

            # Refresh the plot
            plt.draw()
    # Define a function to handle the mouse move event
    def onmove(event):
        # Check if the left mouse button is pressed and a circle patch exists
        if event.button == 1 and circle is not None:
            # Get the x and y coordinates of the mouse move
            x, y = event.xdata, event.ydata

            # Calculate the distance between the mouse click and mouse move
            radius = np.sqrt((x - circle.center[0])**2 + (y - circle.center[1])**2)

            # Update the radius of the circle
            circle.set_radius(radius)

            # Refresh the plot
            plt.draw()
            # np.savetxt("centroid_and_radius.txt", np.array([circle.center[0], circle.center[1], radius]))

    # Define a function to handle the mouse release event
    def onrelease(event):
        # Set the circle patch to None
        global circle
        global speaker
        inside_count = 0
        for i in range(data.shape[0]):
            dist = (circle.center[0] - data[i, 0]) ** 2 + (circle.center[1] - data[i, 1]) ** 2
            dist = np.sqrt(dist)
            if dist < circle.radius:
                inside_count = inside_count + 1
        output.append([circle.center[0], circle.center[1], speaker, inside_count / data.shape[0]])
        radius = circle.radius
        circle.remove()
        if speaker == 1: # if it's a speaker
            new_circle = plt.Circle((output[-1][0], output[-1][1]), radius, color='g', fill=False)
        else:
            new_circle = plt.Circle((output[-1][0], output[-1][1]), radius, color='b', fill=False)
        circle_list.append(new_circle)
        circle = new_circle
        ax.add_patch(circle)
        plt.draw()
        circle = None
        print("[")
        for i in range(0, len(output)):
            print(output[i], ", ")
        print("]")
        print(np.array(output).shape)
        with open(out_path, "wb") as f:
            pkl.dump(np.array(output), f)
    # the undo button
    circle_list = []   
    def on_press(event):
        if event.key == 'z':   
            onundo(event)
        if event.key == "h":
            global speaker
            speaker = speaker * -1


    def onundo(event):
        if circle_list:
            circle = circle_list.pop()
            output.pop()
            circle.remove()
            plt.draw()
            # print(output)

    fig.canvas.mpl_connect('key_press_event', on_press)

    # Connect the mouse click event to the onclick function
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)

    # Connect the mouse move event to the onmove function
    cid2 = fig.canvas.mpl_connect('motion_notify_event', onmove)

    # Connect the mouse release event to the onrelease function
    cid3 = fig.canvas.mpl_connect('button_release_event', onrelease)
    # Display the plot
    plt.show()