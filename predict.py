import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import os
import time
import sys
sys.path.append("object_detection/")
from utils import label_map_util
from utils import visualization_utils as vis_util
PATH_TO_CKPT = os.path.join(os.getcwd(), 'export', 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(os.getcwd(), 'object_detection', 'data', 'cube_label_map.pbtxt')
NUM_CLASSES = 1

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
vids = glob('videos/*.mp4')

# from here: https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np


for (i, vid) in enumerate(vids):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    cap = cv2.VideoCapture(vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output' + str(i) + '.avi', fourcc, 20.0, (width, height))

    #fps tracking
    frame_num = 0
    last_time = time.time()
    smoothing = 0.9
    delta_time = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        #if no frame then break
        if not ret:
            break

        #basically get a smoothed out version of the time between each frame and then take 1/delta_time for fps
        new_time = time.time()
        delta_time = (new_time - last_time) * (1 - smoothing) + (delta_time * smoothing)
        fps = 1/delta_time
        last_time = new_time

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #bounding box using neural network
        frame = detect_objects(frame, sess, detection_graph)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #put fps (can change (0,0,0) to (255, 255, 255) if black text doesn't show up)
        frame = cv2.putText(frame, "{0:.1f}".format(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=3)
        out.write(frame)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    sess.close() 
    