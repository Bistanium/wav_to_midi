import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from math import log10
from wave import open
import numpy as np
from numpy.fft import fft
from scipy.signal import firwin, lfilter

# midi化関数
def data2midi(F: np.ndarray, fs: int, N: int):
    half_n = N // 2
    sec = N / fs
    beforenote, maxvolume = 0, 0
    for i in range(1, half_n):
        i_sec = i/sec # i/secが周波数
        if 64 < i_sec < 11175: # midiの範囲に収める
            # ノート番号計算
            midinote = 69 + log10(i_sec/440)/0.025085832972
            # 音量計算
            volume = (abs(F.imag[i]/N*2) ** (1.8/3)) * 1.125
            if volume > 127: volume = 127

            rounded_midinote = int(round(midinote, 0))
            if not beforenote == rounded_midinote: # 音階が変わったら前の音階をmidiに打ち込む
                rounded_volume = int(round(maxvolume, 0))
                if not rounded_volume == 0:
                    track.append(Message('note_on', note=beforenote, velocity=rounded_volume, time=00))
                beforenote, maxvolume = rounded_midinote, volume
            elif volume > maxvolume: # 同じ音階なら音量を今までの最大値にする
                maxvolume = volume

    lowestnote, highestnote = 36, 126
    for j in range(lowestnote, highestnote):
        soundtime = int(round(120*sec, 0)) if j == lowestnote else 0
        track.append(Message('note_off', note=j, time=soundtime))


# Wave読み込み
def read_wav(file_path: str):
    wf = open(file_path, "rb")

    buf = wf.readframes(-1) # 全部読み込む
    # 16bitごとに10進数化
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype='int16')
    else:
        data = np.zeros(len(buf), dtype=np.complex128)

    # ステレオの場合，チャンネルを分離
    if wf.getnchannels() == 2:
        data_l = data[::2]
        data_r = data[1::2]
    else:
        data_l = data
        data_r = data
    wf.close()
    return data_l, data_r


# wavファイルの情報を取得
def info_wav(file_path: str):
    ret = {}
    wf = open(file_path, "rb")
    ret["ch"] = wf.getnchannels()
    ret["byte"] = wf.getsampwidth()
    ret["fs"] = wf.getframerate()
    ret["N"] = wf.getnframes()
    ret["sec"] = ret["N"] / ret["fs"]
    wf.close()
    return ret


# データ分割
def audio_split(data: np.ndarray, win_size: int):
    splited_data = []
    len_data = len(data)
    win = np.hanning(win_size)
    for i in range(0, len_data, win_size//2):
        endi = i + win_size
        if endi < len_data:
            splited_data.append(data[i:endi] * win)
        else:
            win = np.hanning(len(data[i:-1]))
            splited_data.append(data[i:-1] * win)
    return splited_data


def downsampling(conversion_rate: int, data: np.ndarray, fs: int):
    # FIRフィルタ
    nyqF = fs/2                       # 変換前のナイキスト周波数
    cF = (conversion_rate/2-500)/nyqF # カットオフ周波数
    taps = 511                        # フィルタ係数
    b = firwin(taps, cF)   # LPF
    # フィルタリング
    data = lfilter(b, 1, data)

    # 間引き処理
    downlate = np.arange(0, len(data)-1, fs/conversion_rate)
    rounded_indices = np.round(downlate).astype(int)
    downed_data = data[rounded_indices]
    return downed_data


if __name__ == '__main__':

    # midi定義
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    # テンポ
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(480)))

    # Wav読み込み
    data_l,data_r = read_wav("test.wav")
    del data_r

    # Wavの情報取得
    wi = info_wav("test.wav")

    # ダウンサンプリング
    if wi["fs"] > 40960:
        new_fs = 40960
        downed_data = downsampling(new_fs, data_l, wi["fs"])
    else:
        new_fs = wi["fs"]
        downed_data = data_l
    del data_l

    # ウィンドウサイズ
    win_size = 1024 * 16

    # データ分割
    splited_data = audio_split(downed_data, win_size)
    del downed_data

    # FFT&midi化
    len_splited_data = len(splited_data)
    for i in range(0, len_splited_data):
        ffted_data = fft(splited_data[i])
        data2midi(ffted_data, new_fs, len(ffted_data.imag))

    out_file = "test_wav.mid"
    mid.save(out_file)
