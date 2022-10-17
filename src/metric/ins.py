
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from skimage.transform import resize
from tqdm import tqdm
import numpy as np

#
# def scale_images(images, new_shape):
#     images_list = list()
#     for image in images:
#         # resize with nearest neighbor interpolation
#         new_image = resize(image, new_shape, 0)*255
#         # store
#         images_list.append(new_image)
#     return np.asarray(images_list)
#
#
# def stack_features(image, eval_model, batch_size):
#     N = image.shape[0]
#     act = []
#     for i_ in tqdm(range((N//batch_size)+1), disable=False):
#         # print(i*batch_size, (i+1) * batch_size)
#         batch_ = image[i_*batch_size: (i_+1) * batch_size]
#         batch_ = scale_images(batch_, (299, 299, 3))
#         batch_ = preprocess_input(batch_)
#         act.append(eval_model.predict(batch_))
#     act = np.concatenate(act)
#     return act


def get_inception_score(probs, num_splits=10):
    N = probs.shape[0]
    scores = []

    for s_ in tqdm(range(num_splits), disable=False):
        part = probs[s_ * N // num_splits: (s_ + 1) * N // num_splits, :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), axis=0)))
        kl = np.mean(np.sum(kl, 1))
        kl = np.exp(kl)
        scores.append(np.expand_dims(kl, axis=0))

    scores = np.concatenate(scores, 0)
    m_scores = np.mean(scores)
    std_scores = np.std(scores)
    return m_scores, std_scores


# def stack_feature(eval_model, images):
#     act = []
#     for i in tqdm(range(iter_num)):
#         start = i * batch_size
#         end = i * batch_size + batch_size
#         img_ = real_images[start:end]
#         img_ = img_.astype('float32') / 255.
#         img_ = scale_images(img_, (299, 299, 3))
#         img_ = preprocess_input(img_)
#         act.append(eval_model.predict(img_))
#
#     act = np.concatenate(act)
#     mu, sigma = act.mean(axis=0), np.cov(act, rowvar=False)


# probs = []
# batch_size = 128
# for i, batch in enumerate(loader):
#     print(i+1, len(loader))
#     img, lable = batch[0].type(dtype), batch[1].type(dtype)
#     batch_size_i = img.size(0)
#
#     probs.append(get_pred(img))
# probs = np.concatenate(probs, axis=0)


# m_scores, std_scores = get_inception_score(probs=probs, num_splits=10)