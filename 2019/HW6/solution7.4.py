import numpy as np
from PIL import Image

if __name__ == '__main__':
    image: np.ndarray = np.array(Image.open('Alan_Turing.jpg', 'r'))
    U, sigma, VT = np.linalg.svd(image)
    for k in [2, 4, 8, 16, 32, 64, 128, 256]:
        image_k = np.sum([
            sigma[i] * U[:, i].reshape((512, 1)) @ VT[i, :].reshape((1, 512))
            for i in range(k)
        ], axis=0)
        output = Image.fromarray(image_k).convert('L')
        output.save(f'solution7.4_img/Alan_Turing_k={k}.pdf')
