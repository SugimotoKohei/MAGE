import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


class MAGESE:
    def __init__(self, img, phase_encode=0):
        self.img = img
        self.matrix = img.size
        self.pe = phase_encode # self.pe=0で位相エンコードが左右方向、self.pe=1で位相エンコードが前後方向、
        self.kspace = img2kspace(img)
        self.concate_moved_kspace = []

        self.vertical_moved_image = None
        self.vertical_moved_kspace = None
        self.concate_vertical_moved_kspace = None
        self.horizontal_moved_image = None
        self.horizontal_moved_kspace = None
        self.concate_horizontal_moved_kspace = None
        self.diagonal_moved_image = None
        self.diagonal_moved_kspace = None
        self.concate_diagonal_moved_kspace = None
        self.rotational_moved_image = None
        self.rotational_moved_kspace = None
        self.concate_rotational_moved_kspace = None

    def move_vertical(self, move_pixel):
        move_pixel = np.abs(move_pixel)
        move_range = np.arange(move_pixel*(-1), move_pixel+1, 1)
        length_of_data = len(move_range)

        # 代入するための空の配列を作成
        self.vertical_moved_image = np.zeros([length_of_data, self.matrix[0], self.matrix[1]])
        self.vertical_moved_kspace = np.zeros(self.vertical_moved_image.shape, dtype=np.complex64)

        # データを代入
        for k, v in enumerate(move_range):
            self.vertical_moved_image[k] = self.img.rotate(0, translate=(0, v))
        self.vertical_moved_kspace = img2kspace(self.vertical_moved_image)

        # 各シフトデータのk-spaceからランダムに取り出して、１枚のk-spaceを作成
        self.concate_vertical_moved_kspace = self.create_random_kspace(move_range, self.vertical_moved_kspace)

        # 各方向にシフトした画像を組み合わせた画像の作成用
        self.concate_moved_kspace.append(self.concate_vertical_moved_kspace)

    def move_horizontal(self, move_pixel):
        move_pixel = np.abs(move_pixel)
        move_range = np.arange(move_pixel*(-1), move_pixel+1, 1)
        length_of_data = len(move_range)

        # 代入するための空の配列を作成
        self.horizontal_moved_image = np.zeros([length_of_data, self.matrix[0], self.matrix[1]])
        self.horizontal_moved_kspace = np.zeros(self.horizontal_moved_image.shape, dtype=np.complex64)

        # データを代入
        for k, v in enumerate(move_range):
            self.horizontal_moved_image[k] = self.img.rotate(0, translate=(v, 0))
        self.hprizontal_moved_kspace = img2kspace(self.horizontal_moved_image)

        # 各シフトデータのk-spaceからランダムに取り出して、１枚のk-spaceを作成
        self.concate_horizontal_moved_kspace = self.create_random_kspace(move_range, self.horizontal_moved_kspace)

        # 各方向にシフトした画像を組み合わせた画像の作成用
        self.concate_moved_kspace.append(self.concate_horizontal_moved_kspace)

    def move_diagonal(self, move_pixel):
        move_pixel = np.abs(move_pixel)
        move_range = np.arange(move_pixel*(-1), move_pixel+1, 1)
        length_of_data = 2 * len(move_range)

        # 代入するための空の配列を作成
        self.diagonal_moved_image = np.zeros([length_of_data, self.matrix[0], self.matrix[1]])
        self.diagonal_moved_kspace = np.zeros(self.diagonal_moved_image.shape, dtype=np.complex64)

        # データを代入
        for k, v in enumerate(move_range):
            self.diagonal_moved_image[k] = self.img.rotate(0, translate=(v, v))
        for k, v in enumerate(move_range):
            self.diagonal_moved_image[k + int(len(self.diagonal_move_range))] = self.img.rotate(0, translate=(-v, v))

        # 各シフトデータのk-spaceからランダムに取り出して、１枚のk-spaceを作成
        self.concate_diagonal_moved_kspace = self.create_random_kspace(move_range, self.diagonal_moved_kspace)

        # 各方向にシフトした画像を組み合わせた画像の作成用
        self.concate_moved_kspace.append(self.concate_diagonal_moved_kspace)

    def move_rotational(self, move_pixel):
        move_pixel = np.abs(move_pixel)
        move_range = np.arange(move_pixel*(-1), move_pixel+1, 1)
        length_of_data = len(move_range)

        # 代入するための空の配列を作成
        self.rotational_moved_image = np.zeros([length_of_data, self.matrix[0], self.matrix[1]])
        self.rotational_moved_kspace = np.zeros(self.rotational_moved_image.shape, dtype=np.complex64)

        # データを代入
        for k, v in enumerate(move_range):
            self.rotational_moved_image[k] = self.img.rotate(v)

        # 各シフトデータのk-spaceからランダムに取り出して、１枚のk-spaceを作成
        self.concate_rotational_moved_kspace = self.create_random_kspace(move_range, self.rotational_moved_kspace)

        # 各方向にシフトした画像を組み合わせた画像の作成用
        self.concate_moved_kspace.append(self.concate_rotational_moved_kspace)

    def _get_trunked_normal(self, mean, sd, low, upp, num):
        setting = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        return setting.rvs(num)

    def create_random_kspace(self, move_range, moved_kspace):
        concat_moved_kspace = np.zeros(self.matrix, dtype=np.complex64)

        # 乱数を設定
        mean = np.median(move_range)
        sd = 5
        random_number = self._get_trunked_normal(mean, sd, 0, np.max(move_range), self.matrix[self.pe])
        random_number = np.ceil(random_number).astype(int)
        # random_number = np.random.randint(0, moved_kspace_shape[0]-1, moved_kspace_shape[1])

        # phase_encode=0で位相エンコードが左右方向、phase_encode=1で位相エンコードが前後方向
        if self.pe == 0:
            for key, value in enumerate(random_number):
                concat_moved_kspace[:, key] = moved_kspace[value][:, key]
        elif self.pe == 1:
            for key, value in enumerate(random_number):
                concat_moved_kspace[key, :] = moved_kspace[value][key, :]

        return concat_moved_kspace

    def create_multiple_direction_move_kspace(self):
        self.multiple_moved_kspace = np.zeros(self.matrix, dtype=np.complex64)

        # phase_encode=0で位相エンコードが左右方向、phase_encode=1で位相エンコードが前後方向
        if self.pe == 0:
            for i in range(self.matrix[self.pe]):
                random_number = np.random.randoint(0, len(self.concate_moved_kspace))
                self.multiple_moved_kspace[:, i] =  self.concate_moved_kspace[random_number][:, i]
        elif self.pe == 1:
            for i in range(self.matrix[self.pe]):
                random_number = np.random.randoint(0, len(self.concate_moved_kspace))
                self.multiple_moved_kspace[i, :] =  self.concate_moved_kspace[random_number][i, :]

class MAGETSE(MAGESE):
    def __init__(self, img, etl, phase_encode=0):
        super().__init__(img, phase_encode)
        self.etl = etl

    def create_random_kspace(self, move_range, moved_kspace):
        concat_moved_kspace = np.zeros(self.matrix, dtype=np.complex64)

       # 乱数を設定
        mean = np.median(move_range)
        sd = 2
        random_number = self._get_trunked_normal(mean, sd, 0, np.max(move_range), int(self.matrix[self.pe]/self.etl))
        random_number = np.ceil(random_number).astype(int)
        # random_number = np.random.randint(0, moved_kspace_shape[0]-1, moved_kspace_shape[1])

        # phase_encode=0で位相エンコードが左右方向、phase_encode=1で位相エンコードが前後方向
        if self.pe == 0:
            for key, value in enumerate(random_number):
                for e in range(self.etl):
                    concat_moved_kspace[:, key*(e+1)] = moved_kspace[value][:, key*(e+1)]
        elif self.pe == 1:
            for key, value in enumerate(random_number):
                for e in range(self.etl):
                    concat_moved_kspace[key*(e+1), :] = moved_kspace[value][key*(e+1), :]

        return concat_moved_kspace


def img2kspace(img):
    kspace = np.fft.fftshift(np.fft.fft2(img))
    return kspace
    
def kspace2img(kspace):
    img = np.fft.ifft2(np.fft.ifftshift(kspace))
    return img

def plot_kspace(kspace):
    adjust_kspace = 20 * np.log(np.abs(kspace))
    plt.imshow(adjust_kspace, cmap='gray')
    plt.show()