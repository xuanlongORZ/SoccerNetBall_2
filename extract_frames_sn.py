import os
import argparse
import cv2
from moviepy import *
from tqdm import tqdm
from multiprocessing import Pool
cv2.setNumThreads(0)
from SoccerNet.Downloader import getListGames
from util.io import load_text, load_json

'''
This script extracts frames from SoccerNetv2 Action Spotting dataset by introducing the path where the downloaded videos are (at 720p resolution), the path to
write the frames, the sample fps, and the number of workers to use. The script will create a folder for each video in the out_dir path and save the frames as .jpg files in
the desired resolution. We only download frames around ground-truth actions, as we only use clips with actions while training, and we want to reduce the size of the extracted frames.

python extract_frames_sn.py --video_dir video_dir
        --out_dir out_dir
        --sample_fps 12.5 --num_workers 5
'''



FRAME_CORRECT_THRESHOLD = 1000
SPLIT = ['train', 'valid']
GT_RADI = 6 # 6 seconds around the ground-truth action (should be enough for clips of 8 seconds)
SN_FPS = 25


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Path to the downloaded videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--sample_fps', type=float, default=12.5)
    parser.add_argument('-j', '--num_workers', type=int,
                        default=os.cpu_count() // 4)
    parser.add_argument('--target_height', type=int, default=448)
    parser.add_argument('--target_width', type=int, default=796)
    parser.add_argument('--original_resolution', type=str, default='720p')
    return parser.parse_args()


def get_duration(video_path):
    return moviepy.editor.VideoFileClip(video_path).duration


def worker(args):

    video_name, video_path, out_dir, sample_fps = args

    LABELS_SN_PATH = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
    label_path = os.path.join(LABELS_SN_PATH, "/".join(video_name.split('/')[:-1]) + '/Labels-v2.json')
    labels_file = load_json(label_path)['annotations']
    half = int(video_name.split('/')[-1][0])

    #Initialize action information with the first action (index, frame, and half)
    act_idx = 0
    act_frame = int(int(labels_file[act_idx]['position']) / 1000 * SN_FPS)
    act_half = int(labels_file[act_idx]['gameTime'][0])

    if act_half != half:
        while act_half != half:
            act_idx += 1
            act_frame = int(int(labels_file[act_idx]['position']) / 1000 * SN_FPS)
            act_half = int(labels_file[act_idx]['gameTime'][0])

    def get_stride(src_fps):
        if sample_fps <= 0:
            stride = 1
        else:
            stride = int(src_fps / sample_fps)
        return stride

    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)

    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    oh = TARGET_HEIGHT
    ow = TARGET_WIDTH

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #Double pass to check if the video is corrupted (different nº of frames as expected)

    #1st pass check nº of frames
    nframes = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        nframes += 1

    vc.release()

    vc = cv2.VideoCapture(video_path)

    #Adapt the fps to the real number of frames
    if num_frames - nframes > FRAME_CORRECT_THRESHOLD:
        effective_fps = fps * nframes / num_frames
        print('Not aligned frames, modified effective fps:', effective_fps)
    else:
        effective_fps = fps
    
    
    #2nd pass to extract frames (if the video is not corrupted)
    if effective_fps == fps:
        not_done = True
        stride = get_stride(fps)
        while not_done:
            est_out_fps = fps / stride
            print('{} -- effective fps: {} (stride: {})'.format(
                video_name, est_out_fps, stride))

            i = 0
            while True:
                ret, frame = vc.read()
                if not ret:
                    # case when the video is corrupted
                    if i != num_frames:
                        print('Failed to decode: {} -- {} / {}'.format(
                            video_path, i, num_frames))

                        if i + FRAME_CORRECT_THRESHOLD < num_frames:
                            print('Difference between expected and actual number of frames is too large')
                            
                    not_done = False
                    break

                if i % stride == 0:
                    if (i >= act_frame - GT_RADI * SN_FPS) & (i <= act_frame + GT_RADI * SN_FPS):
                        if frame.shape[0] != oh or frame.shape[1] != ow:
                            frame = cv2.resize(frame, (ow, oh))
                        if out_dir is not None:
                            frame_path = os.path.join(
                                out_dir, 'frame{}.jpg'.format(i))
                            cv2.imwrite(frame_path, frame)
                    if i >= act_frame + GT_RADI * SN_FPS:
                        act_idx += 1
                        if act_idx < len(labels_file):
                            act_frame = int(int(labels_file[act_idx]['position']) / 1000 * SN_FPS)
                            act_half = int(labels_file[act_idx]['gameTime'][0])
                        else:
                            print('Iterated over all actions in the half')
                            not_done = False
                            break
                        if act_half != half:
                            print('Iterated over all actions in the half')
                            not_done = False
                            break
                i += 1
        vc.release()

        print('{} - done'.format(video_name))

    #2nd pass to extract frames (if the video is corrupted)
    else:
        print('Video {} with strange framerate (effective fps: {})'.format(video_name, effective_fps))
        not_done = True
        stride = get_stride(fps)
        while not_done:
            print('{} -- effective fps: {} (stride: {})'.format(
                    video_name, effective_fps, stride))

            out_frame_num = 0
            i = 0
            while True:
                ret, frame = vc.read()
                if not ret:
                    # fps and num_frames are wrong
                    print('finished with total numer of frames: ' + str(out_frame_num))
                    not_done = False
                    break
                    
                aux_i = i * fps / effective_fps
                if aux_i > out_frame_num:
                    if (out_frame_num >= act_frame - GT_RADI * SN_FPS) & (out_frame_num <= act_frame + GT_RADI * SN_FPS):
                        if frame.shape[0] != oh or frame.shape[1] != ow:
                            frame = cv2.resize(frame, (ow, oh))

                        if out_dir is not None:
                            frame_path = os.path.join(
                                out_dir, 'frame{}.jpg'.format(out_frame_num))
                            cv2.imwrite(frame_path, frame)
                    if out_frame_num >= act_frame + GT_RADI * SN_FPS:
                        act_idx += 1
                        if act_idx < len(labels_file):
                            act_frame = int(int(labels_file[act_idx]['position']) / 1000 * SN_FPS)
                            act_half = int(labels_file[act_idx]['gameTime'][0])
                        else:
                            print('Iterated over all actions in the half')
                            not_done = False
                            break
                        if act_half != half:
                            print('Iterated over all actions in the half')
                            not_done = False
                            break
                    out_frame_num += stride
                i += 1

    vc.release()

    print('{} - done'.format(video_name))

def main(args, games = None):
    video_dir = args.video_dir
    out_dir = args.out_dir
    sample_fps = args.sample_fps
    num_workers = args.num_workers
    target_height = args.target_height
    target_width = args.target_width
    original_resolution = args.original_resolution

    global TARGET_HEIGHT
    TARGET_HEIGHT = target_height
    global TARGET_WIDTH
    TARGET_WIDTH = target_width

    out_dir = out_dir + str(TARGET_HEIGHT)

    worker_args = []

    for game in games:
        for video_file in os.listdir(os.path.join(video_dir, game)):
            
            if (video_file.endswith(original_resolution + '.mkv') | video_file.endswith(original_resolution + '.mp4')):
                half = os.path.splitext(video_file)[0].replace(
                    '_720p', '')
                worker_args.append((os.path.join(game, video_file), 
                                    os.path.join(video_dir, game, video_file),
                                    os.path.join(out_dir, game, 'half' + str(half)),
                                    sample_fps))

    # num_workers = 1
    with Pool(num_workers) as p:
        for _ in tqdm(p.imap_unordered(worker, worker_args),
                    total = len(worker_args)):
            pass

    print('Done!')


if __name__ == '__main__':
    args = get_args()
    games = getListGames(SPLIT)
    main(args, games)