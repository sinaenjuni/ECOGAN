
import numpy as np
from skimage.transform import resize
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from scipy import linalg
from tqdm import tqdm

def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)*255
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

def stack_features(image, eval_model, batch_size):
    N = image.shape[0]
    act = []
    for i_ in tqdm(range((N//batch_size)+1), disable=False):
        # print(i*batch_size, (i+1) * batch_size)
        batch_ = image[i_*batch_size: (i_+1) * batch_size]
        batch_ = scale_images(batch_, (299, 299, 3))
        batch_ = preprocess_input(batch_)
        act.append(eval_model.predict(batch_))
    act = np.concatenate(act)
    mu, sigma = act.mean(axis=0), np.cov(act, rowvar=False)
    return mu, sigma



# def get_real_data_mu_n_sigma(real_images, eval_model,iter_num, batch_size):
#     # eval_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
#
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
#
#     return mu, sigma
#
#
# def get_fake_data_mu_n_sigma(gen_path, eval_model, data_len, iter_num, batch_size):
#     generator = load_model(gen_path)
#     # eval_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
#
#     act = []
#     for i in tqdm(range(iter_num)):
#         label = np.random.uniform(0, 10, batch_size).astype(np.long)
#         noise = np.random.normal(0, 1, (batch_size, generator.input_shape[0][1]))
#         gen_samples_ = generator.predict([noise, label])
#         gen_samples_ = gen_samples_ * 0.5 + 0.5
#         gen_samples_ = scale_images(gen_samples_, (299, 299, 3))
#         gen_samples_ = preprocess_input(gen_samples_)
#         act.append(eval_model.predict(gen_samples_))
#
#     act = np.concatenate(act)[:data_len]
#     mu, sigma = act.mean(axis=0), np.cov(act, rowvar=False)
#     return mu, sigma


def get_fid_score(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6):
    mu_real = np.atleast_1d(mu_real)
    mu_fake = np.atleast_1d(mu_fake)

    sigma_real = np.atleast_2d(sigma_real)
    sigma_fake = np.atleast_2d(sigma_fake)

    assert mu_real.shape == mu_fake.shape, \
        "Training and test mean vectors have different lengths."
    assert sigma_real.shape == sigma_fake.shape, \
        "Training and test covariances have different dimensions."

    diff = mu_real - mu_fake

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * tr_covmean)