import tensorflow as tf

from object_detection.utils import dataset_util
import glob
import xml.etree.ElementTree as ET

flags = tf.app.flags
#Configure path here
flags.DEFINE_string('output_path', 'object_detection/data/evallabels.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(file):
    #use xml to process xml annotations
    tree = ET.parse(file)
    root = tree.getroot()
    height = int(root.find("./size/height").text) # Image height
    width = int(root.find("./size/width").text) # Image width
    filename = root.find("./filename").text # Filename of the image. Empty if image is not from file

    #print(type(height), type(width), type(filename))
    #Configure path here
    with tf.gfile.GFile("EvalImages/" + filename, 'rb') as fid:
        encoded_image_data = fid.read()  # Encoded image bytes

    

    image_format = b'png' if ("png" in filename.lower()) else b'jpeg' # b'jpeg' or b'png'

    filename = filename.encode('utf-8')
    #print(type(image_format), type(filename))
    #print(height, width, filename, image_format)

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for obj in root.findall("./object"):
        classes_text.append("cube".encode('utf-8'))
        classes.append(1)
        xmins.append(float(int(obj.find("bndbox/xmin").text)/width))
        xmaxs.append(float(int(obj.find("bndbox/xmax").text)/width))
        ymins.append(float(int(obj.find("bndbox/ymin").text)/height))
        ymaxs.append(float(int(obj.find("bndbox/ymax").text)/height))

    #print(xmins, xmaxs, ymins, ymaxs, classes_text, classes)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # TODO(user): Write code to read in your dataset to examples variable
    #Configure path here
    files = glob.glob("evalannotations/*.xml")
    #print(files)

    for file in files:
        print(file)
        tf_example = create_tf_example(file)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
  tf.app.run()