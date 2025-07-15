import cv2
import numpy as np
import os
import argparse


class CameraCalibrator:
    """
    カメラキャリブレーションを行うクラス
    """

    def __init__(self, cols, rows, square_size, image_dir):
        """
        Parameters:
            cols (int): チェスボードの列数（内側の交点数）
            rows (int): チェスボードの行数（内側の交点数）
            square_size (float): マスの1辺の実際の長さ（cm）
            image_dir: キャリブレーション画像のフォルダパス
        """

        # チェスボードのサイズとマスのサイズを設定
        self.chessboard_size = (cols, rows)
        self.square_size = square_size
        # キャリブレーション画像のディレクトリ
        self.image_dir = image_dir

        # チェスボードの3D座標を生成
        objp = np.zeros((cols * rows, 3), np.float32)  # 3D座標を格納する配列
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # XYに正規化(Zを削除)
        objp *= square_size  # スケーリング
        self.objp = objp  # 3D 座標群

        # 3D座標と2D座標を格納するリスト
        self.objpoints = []
        self.imgpoints = []

        # 内部パラメータ
        self.camera_matrix = None  # カメラ行列
        self.dist_coeffs = None  # 歪み係数

        # 外部パラメータ
        self.rvec = None  # 回転ベクトル
        self.tvec = None  # 平行移動ベクトル
        self.R = None  # 回転行列

        # self.result_image = None  # 軸を描画した画像

    def calibrate(self):
        """
        カメラキャリブレーションを実行する関数（内部パラメータを求める）
        """

        # キャリブレーション画像の読み込み
        for image_name in os.listdir(self.image_dir):
            img_path = os.path.join(self.image_dir, image_name)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # チェスボード検出
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                # 各画像の3次元座標と2次元座標を追加
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

                # # コーナー描画（確認用）
                # cv2.drawChessboardCorners(img, self.chessboard_size, corners, ret)
                # cv2.imshow("Corners", img)
                # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # キャリブレーション
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,  # 3次元座標群（ワールド座標系）
            self.imgpoints,  # 2次元座標群（ピクセル座標）
            gray.shape[::-1],  # 画像サイズ（幅、高さ）
            None,  # 初期のカメラ行列（Noneなら自動推定）
            None,  # 初期の歪み係数（Noneなら自動推定）
        )
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs

    def estimate_pose(self, input_img):
        """
        カメラ姿勢推定を行う関数
        Parameters:
            input_img: カメラ姿勢推定を行いたい画像
        """

        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        # チェスボード検出
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        if ret:
            success, rvec, tvec = cv2.solvePnP(
                self.objp, corners, self.camera_matrix, self.dist_coeffs
            )

            if success:
                # 回転行列に変換
                R, _ = cv2.Rodrigues(rvec)

                self.rvec = rvec
                self.tvec = tvec
                self.R = R
                self.result_image = input_img.copy()
        else:
            print(
                "チェスボードのコーナーが検出できませんでした。画像を確認してください。"
            )

    def save_dat(self, save_dat_dir):
        """
        キャリブレーション結果を.datファイルに保存する関数
        Parameters:
            save_dat_dir: キャリブレーション結果（.datファイル）を保存するフォルダパス
        """

        if (
            self.camera_matrix is not None
            and self.R is not None
            and self.tvec is not None
        ):
            dat_path = os.path.join(save_dat_dir, "K.dat")
            with open(dat_path, "w") as f:
                np.savetxt(f, self.camera_matrix, fmt="%.6f")
                print("内部パラメータを保存しました:", dat_path)

            dat_path = os.path.join(save_dat_dir, "R.dat")
            with open(dat_path, "w") as f:
                np.savetxt(f, self.R, fmt="%.6f")
                print("回転行列を保存しました:", dat_path)

            dat_path = os.path.join(save_dat_dir, "t.dat")
            with open(dat_path, "w") as f:
                np.savetxt(f, self.tvec, fmt="%.6f")
                print("並進ベクトルを保存しました:", dat_path)

    def draw_axis(self, length=3.0, thickness=3):
        """
        カメラ座標系における座標軸 (X:赤, Y:緑, Z:青) を描画する
        Parameters:
            length (float): 描画する座標軸の長さ（cm）
            thickness (int): 描画する座標軸の太さ
        Returns:
            (bool, np.ndarray): 成功したかどうかと描画された画像
        """

        # 座標軸の長さと太さ
        self.length = length
        self.thickness = thickness

        # 座標軸の3D座標（原点、X軸先端、Y軸先端、Z軸先端）
        axis_3d = np.float32(
            [
                [0, 0, 0],  # 原点
                [self.length, 0, 0],  # X軸
                [0, self.length, 0],  # Y軸
                [0, 0, self.length],  # Z軸
            ]
        )

        if (
            self.camera_matrix is not None
            and self.dist_coeffs is not None
            and self.rvec is not None
            and self.tvec is not None
        ):
            # 2D画像上の座標に投影
            imgpts, _ = cv2.projectPoints(
                axis_3d, self.rvec, self.tvec, self.camera_matrix, self.dist_coeffs
            )

            # 座標をintに変換
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # 線を描画
            origin = imgpts[0]
            cv2.line(
                self.result_image, origin, imgpts[1], (0, 0, 255), self.thickness
            )  # X軸（赤）
            cv2.line(
                self.result_image, origin, imgpts[2], (0, 255, 0), self.thickness
            )  # Y軸（緑）
            cv2.line(
                self.result_image, origin, imgpts[3], (255, 0, 0), self.thickness
            )  # Z軸（青）

            return (True, self.result_image)
        return (False, None)  # カメラパラメータが未設定の場合はNoneを返す


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    # 以下、必須入力
    parser.add_argument(
        "-c",
        "--cols",
        type=int,
        required=True,
        help="Number of inner corners (columns)",
    )
    parser.add_argument(
        "-r", "--rows", type=int, required=True, help="Number of inner corners (rows)"
    )
    parser.add_argument(
        "-s",
        "--square_size",
        type=float,
        required=True,
        help="Size of one square (e.g. in cm)",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        required=True,
        help="Directory of calibration images",
    )
    parser.add_argument(
        "-o",
        "--save_dat_dir",
        type=str,
        required=True,
        help="Path to save calibration result (.dat files)",
    )

    # 以下、オプション入力
    parser.add_argument(
        "--length",
        type=float,
        default=3.0,
        help="Length of the axis to draw (default: 5.0 cm)",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Thickness of the axis lines (default: 2)",
    )

    args = parser.parse_args()

    calibrator = CameraCalibrator(
        # 必須引数を渡す
        cols=args.cols,
        rows=args.rows,
        square_size=args.square_size,
        image_dir=args.image_dir,
    )

    # キャリブレーションを実行（内部パラメータを求める）
    calibrator.calibrate()
    print("カメラ内部パラメータ:\n", calibrator.camera_matrix)
    print("歪み係数:\n", calibrator.dist_coeffs)

    # カメラの姿勢推定を実行
    img = cv2.imread("./input.jpg")
    calibrator.estimate_pose(img)
    print("回転ベクトル (rvec):\n", calibrator.rvec)
    print("並進ベクトル (tvec):\n", calibrator.tvec)
    print("回転行列 (R):\n", calibrator.R)

    # キャリブレーション結果を.datファイルに保存
    calibrator.save_dat(args.save_dat_dir)

    # 座標軸を描画して保存
    ret, img = calibrator.draw_axis(args.length, args.thickness)
    if ret:
        cv2.imwrite("./output.jpg", img)
