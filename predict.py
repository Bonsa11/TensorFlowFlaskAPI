from tensorflow import keras

from image_pre_processing import (process_fundus_retinal_1, process_fundus_retinal_2,
                                  process_fundus_retinals_1, process_fundus_retinals_2)

min_class_bound = -4  # classification bounds for anterior images
max_class_bound = 11

model_1 = keras.models.load_model('./models/classification_1_model.h5')
model_2 = keras.models.load_model('./models/classification_2_model.h5')


def predict_fundus_class(img_64):
    img = process_fundus_retinal_1(img_64)
    prediction_1 = float(model_1.predict(img))

    img = process_fundus_retinal_2(img_64)
    prediction_2 = model_2.predict(img)[0]

    return prediction_1, prediction_2


def classify_fundus_class(img_64):
    img = process_fundus_retinal_1(img_64)
    prediction = model_1.predict(img)
    if float(prediction) < min_class_bound:
        return 0
    elif float(prediction) > max_class_bound:
        return 1
    else:
        img = process_fundus_retinal_2(img_64)
        prediction = model_2.predict(img)
        print(prediction)

        if prediction[0][0] + prediction[0][1] > 1.1:
            return -1
        elif prediction[0][0] + prediction[0][1] < 0.4:
            return -1
        elif prediction[0][0] > 0.996 and prediction[0][1] < 0.005:
            return 0
        elif prediction[0][1] > 0.93:
            return 1
        else:
            return -1


def multi_predict_fundus_class(files: dict):
    filenames = [data['filename'] for data in files.values()]
    images = [data['image'] for data in files.values()]
    results = {}
    try:
        processed_imgs = process_fundus_retinals_1(images)
        predictions = model_1.predict(processed_imgs)
        for filename, prediction in zip(filenames, predictions):
            results[filename] = {'model_1' : str(prediction),
                                'model_2' : 'NA'}

    except Exception as e:
        print(f'failed to predict model1 : {e}')

    try:
        iter_processed_imgs = process_fundus_retinals_2(images)
        iter_predictions = model_2.predict(iter_processed_imgs)
        for filename, prediction in zip(filenames, iter_predictions):
            results[filename]['model_2'] = str(prediction)
    except Exception as e:
        print(f'failed to predict model2 : {e}')

    return results


def multi_classify_fundus_class(files: dict):
    filenames = [data['filename'] for data in files.values()]
    images = [data['image'] for data in files.values()]
    results = {}
    try:
        processed_imgs = process_fundus_retinals_1(images)
        predictions = model_1.predict(processed_imgs)
        for filename, prediction in zip(filenames,predictions):
            try:
                if float(prediction) < min_class_bound:
                    results[filename] = 0
                elif float(prediction) > max_class_bound:
                    results[filename] = 1
                else:
                    results[filename] = -1
            except Exception as e:
                results[filename] = -1
    except Exception as e:
        print(f'failed to predict model1 : {e}')

    try:
        iter_filenames = [filename for filename in filenames if results[filename] == -1]
        iter_images = [images[filenames.index(filename)] for filename in iter_filenames]
        iter_processed_imgs = process_fundus_retinals_2(iter_images)
        iter_predictions = model_2.predict(iter_processed_imgs)
        for filename, prediction in zip(iter_filenames, iter_predictions):
            try:
                if prediction[0][0] + prediction[0][1] > 1.1:
                    results[filename] = -1
                elif prediction[0][0] + prediction[0][1] < 0.4:
                    results[filename] = -1
                elif prediction[0][0] > 0.996 and prediction[0][1] < 0.005:
                    results[filename] = 0
                elif prediction[0][1] > 0.93:
                    results[filename] = 1
                else:
                    results[filename] = -1
            except Exception as e:
                results[filename] = -1
    except Exception as e:
        print(f'failed to predict model2 : {e}')

    return results