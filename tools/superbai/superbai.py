import valohai
import zipfile
import tempfile
import glob
import json
import os
import tensorflow as tf


inputs = {
    "images": ".data/data_coco.zip",
    "labels": ".data/labels_coco.zip",
}

valohai.prepare(step="convert-superbai", default_inputs=inputs)


def main():
    images_path = valohai.inputs('images').path(process_archives=False)
    labels_path = valohai.inputs('labels').path(process_archives=False)

    with tempfile.TemporaryDirectory() as labels_temp_folder, tempfile.TemporaryDirectory() as images_temp_folder:
        with zipfile.ZipFile(images_path, 'r') as images_zip, zipfile.ZipFile(labels_path, 'r') as labels_zip:
            images_zip.extractall(images_temp_folder)
            labels_zip.extractall(labels_temp_folder)

        with open(os.path.join(labels_temp_folder, 'project.json')) as json_file:
            json_data = json.load(json_file)
            version = json_data["version"] if "version" in json_data else "legacy"

            if version == "legacy":
                classes = [obj['class_name'] for obj in json_data['objects'] if 'box' in obj['info']['shapes']]
            else:
                classes = [oclass['name'] for oclass in json_data['object_detection']['object_classes'] if 'box' in oclass['annotation_type'] == 'box']

            classes_path = valohai.outputs().path(filename='classes.txt')
            with open(classes_path, mode='w') as myfile:
                myfile.write('\n'.join(classes))
            print(f"Class mapping loaded: {classes}")
            print(f"Wrote {classes_path}")

        for file_path in glob.glob(os.path.join(images_temp_folder, '**/*.jpg')):
            # Remove project folder and extra .jpg.jpg (currently how SuperbAI exports)
            # Neither are explicitly readable from the JSON
            basename = os.path.basename(file_path)
            fixed_basename = basename.replace('.jpg', '') + '.jpg'
            os.rename(file_path, os.path.join(images_temp_folder, os.path.basename(fixed_basename)))

        print("Parsing images...")
        training_path = valohai.outputs('train').path('train.tfrecord')
        test_path = valohai.outputs('test').path('test.tfrecord')
        with tf.io.TFRecordWriter(training_path) as training_writer, tf.io.TFRecordWriter(test_path) as test_writer:
            for file_path in glob.glob(os.path.join(labels_temp_folder, 'meta/**/*.json')):
                with open(file_path) as json_file:
                    json_data = json.load(json_file)
                    if version == "legacy":
                        import superbai_converter_legacy
                        tf_record, test = superbai_converter_legacy.parse_labels_to_tensorflow(json_data, classes, images_temp_folder, labels_temp_folder)
                    else:
                        import superbai_converter_current
                        tf_record, test = superbai_converter_current.parse_labels_to_tensorflow(json_data, classes, images_temp_folder, labels_temp_folder)

                    if tf_record:
                        writer = test_writer if test else training_writer
                        writer.write(tf_record.SerializeToString())
        print(f"Wrote training data {training_path}")
        print(f"Wrote test data {test_path}")
        print("Done.")


if __name__ == "__main__":
    main()
