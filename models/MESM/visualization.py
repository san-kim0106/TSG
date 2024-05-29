import os
import random
import numpy as np
import pandas as pd
import json, jsonlines
import logging
from tqdm import tqdm
getattr(tqdm, '_instances', {}).clear()
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300
import matplotlib.pyplot as plt
#from IPython.display import Video, HTML

# for visualization
import cv2
from moviepy.editor import *

from utils.func_utils import load_json
from dataset.charades import CharadesDataset

np.set_printoptions(precision=3, suppress=True)

logging.basicConfig(filename='visualization/charades/_error.log', level=logging.ERROR)

#----------------------------------------------------------------------------------------#

plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)

#----------------------------------------------------------------------------------------#

from PIL import Image, ImageDraw, ImageFont

FAST = True

def expand_mask(mask, margin=2, height=12):
    w = mask.shape[1]
    out = [np.zeros((1,w,3), dtype=np.int_) for i in range(margin)]
    for i in range(height):
        out.append(mask)
    for i in range(margin):
        out.append(np.zeros((1,w,3), dtype=np.int_))
    return np.concatenate(out, axis=0)

def text_phantom(text, width=480):
    # Availability is platform dependent
    # font = 'DejaVuSans-Bold'
    font = 'Lato-Bold'

    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=16,
                                  encoding="unic")
    text_width, text_height = pil_font.getsize(text)

    # create a blank canvas with extra space between lines
    canvas = Image.new('RGB', [width,20], (255, 255, 255))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    white = "#000000"
    draw.text((0,0), text, font=pil_font, fill=white)

    # (text, background): (black, while) -> (white, black)
    return 255 - np.asarray(canvas)

def sampling_idx(preds, policy="random"):
    idx = random.randint(0, len(preds["gts"])-1)
    if policy == "random":
        return idx
    elif policy == "success":
        pred = preds["predictions"][idx][0]
        gt = preds["gts"][idx]
        while compute_tiou(pred, gt) < 0.8 or preds["gts"][idx][0] < 15:
            idx = random.randint(0, len(preds["gts"])-1)
            pred = preds["predictions"][idx][0]
            gt = preds["gts"][idx]
    elif policy == "failure":
        pred = preds["predictions"][idx][0]
        gt = preds["gts"][idx]
        while compute_tiou(pred, gt) > 0.2:
            idx = random.randint(0, len(preds["gts"])-1)
            pred = preds["predictions"][idx][0]
            gt = preds["gts"][idx]
    return idx

#----------------------------------------------------------------------------------------#

def make_bar(gt, pred, vlen, wbar):
    # draw bar for GT and prediction
    gt_idx = np.round(np.asarray(gt) / vlen * wbar).astype(np.int_)
    pred_idx = np.round(np.asarray(pred) / vlen * wbar).astype(np.int_)
    gt_mask, pred_mask = np.zeros((1,wbar,3)), np.zeros((1,wbar,3))

    gt_mask[0, gt_idx[0]:gt_idx[1], 0] = 255 # Red color
    pred_mask[0, pred_idx[0]:pred_idx[1], 2] = 255 # blue color

    # expand masks for better visualization and concatenate them
    bar = np.concatenate([expand_mask(gt_mask, margin=4), expand_mask(pred_mask)], axis=0)
    return bar

def make_result_video(preds, D, dt, vid_dir, policy="random", visualize=True):
    # sampling index and fetching relevant information
    #policy = "success" # among ["random" | "success" | "failure"]

    # video_df = pd.DataFrame(columns=["video_id", "query", "IoU", "failed_to_load"])
    
    for idx in tqdm(range(len(preds["gts"])), desc="Video visualization"):
      qid = preds["qids"][idx]
      pred = preds["predictions"][idx]
      gt = preds["gts"][idx]
      vid = preds["vids"][idx]
      query = preds["query"][idx]
      # assert vid == D.anns[qid]["video_id"], "{} != {}".format(vid, D.anns[qid]["video_id"])
      # assert vlen == D.anns[qid]["duration"], "{} != {}".format(vlen, D.anns[qid]["duration"])
      tiou = compute_tiou(pred, gt)

      # concatenate two videos where one for GT (red) and another for prediction (blue)
      vw, mg, nw = 480, 20, 50 # video_width, margin, number of words at each line
      if dt == "anet":
          vname = vid[2:] + ".mp4"
      elif dt == "charades":
          vname = vid + ".mp4"
      else:
          raise NotImplementedError()
      vid_path = vid_dir + vname
      print(f"video path: {vid_path}")

      try:
        _v = VideoFileClip(vid_path)
        vlen = _v.duration
        _v.close()

        vid_GT = concatenate_videoclips([
            VideoFileClip(vid_path).subclip(0, gt[0]).margin(mg),
            VideoFileClip(vid_path).subclip(gt[0], min(gt[1],vlen)).margin(mg, color=(255,0,0)), # red
            VideoFileClip(vid_path).subclip(min(gt[1],vlen), vlen).margin(mg),
            ])
        vid_pred = concatenate_videoclips([
            VideoFileClip(vid_path).subclip(0, pred[0]).margin(mg),
            VideoFileClip(vid_path).subclip(pred[0], min(pred[1],vlen)).margin(mg, color=(0,0,255)), # blue
            VideoFileClip(vid_path).subclip(min(pred[1],vlen), vlen).margin(mg),
            ])
        cc = clips_array([[vid_GT, vid_pred]]).resize(width=vw)
        
        if FAST:
            if dt == "charades":
                factor = np.round(vlen / 20)
            else:
                factor = np.round(vlen / 30)
            print(f"speedup factor: {factor}")
            cc = cc.speedx(factor=factor)

        print(f"duration  : {vlen}")
        print(f"vid       : {vid}")
        print(f"Q         : {query}")
        print(f"prediction: {pred}")
        print(f"gt.       : {gt}")
        #cc.ipython_display(width=vw, maxduration=300)
        #cc.ipython_display(maxduration=300)

        # draw query in text image
        query = "Q: " + query + " | IoU: " + str(tiou)
        nlines = np.int_(np.ceil(len(query) / nw))
        tline = []
        for nl in range(nlines):
            if nl == nlines-1:
                cur_text = text_phantom(query[nl*nw:], vw)
            else:
                cur_text = text_phantom(query[nl*nw:nl*nw+nw], vw)
            tline.append(cur_text)
        q_text = np.concatenate(tline, axis=0)

        # draw bar for GT and prediction
        gt_idx = np.round(np.asarray(gt) / vlen * vw).astype(np.int_)
        pred_idx = np.round(np.asarray(pred) / vlen * vw).astype(np.int_)
        gt_mask, pred_mask = np.zeros((1,vw,3)), np.zeros((1,vw,3))
        gt_mask[0, gt_idx[0]:gt_idx[1], 0] = 255 # Red color
        pred_mask[0, pred_idx[0]:pred_idx[1], 2] = 255 # blue color
        # expand masks for better visualization and concatenate them
        bar = np.concatenate([expand_mask(gt_mask, margin=4), expand_mask(pred_mask)], axis=0)

        # video_df.loc[len(video_df.index)] = [vid, query, tiou, 0]
        
        def add_query_and_bar(frame):
            """ Add GT/prediction bar into frame"""
            return np.concatenate([q_text, frame, bar], axis=0)

        final_clip = cc.fl_image(add_query_and_bar)
        
        if visualize:
            final_clip.ipython_display(maxduration=300)
        else:
            os.makedirs(f"visualization/{dt}", exist_ok=True)
            save_to = f"visualization/{dt}/{vid}_{str(tiou)}.mp4"
            final_clip.write_videofile(save_to, fps=final_clip.fps)
        
        vid_GT.close()
        vid_pred.close()
        
        print("Visualization saved!!!")
      except Exception as e:
          # video_df.loc[len(video_df.index)] = [vid, query, tiou, 1]
          logging.exception(f"An exception occurred during handling path: {vid_path}: {str(e)}\n\n")

    # video_df.to_csv(f"visualization/{dt}/video_df.csv", index=False)

#----------------------------------------------------------------------------------------#

def load_result(pred_path):
    results = []
    with jsonlines.open(pred_path, "r") as f:
        for line in f.iter():
            results.append(line)
    
    
    gts = pd.DataFrame(columns=["vid", "start", "end", "query"])
    with open("data/charades/annotations/charades_sta_test.txt", "r") as f:
        for line in f.readlines():
            _, _query = line.split("##")
            _vid, _start, _end = _.split(" ")
            gts.loc[len(gts.index)] = [_vid.strip(), float(_start.strip()), float(_end.strip()), _query.strip()]
    
    preds = {"qids": [], "predictions": [], "gts": [], "vids": [], "query": []}

    for result in results:
        preds["qids"].append(result["qid"])
        preds["predictions"].append(result["pred_relevant_windows"][0][:2])

        gt = gts[(gts["vid"] == result["vid"]) & (gts["query"] == result["query"])]
        preds["gts"].append([list(gt["start"])[0], list(gt["end"])[0]])

        preds["vids"].append(result["vid"])

        preds["query"].append(result["query"])

    # print(preds)

    return preds

    

def load_output(dt):
    if dt == "anet":
        pass
        
    elif dt == "charades":
        pred_path = "results/eval/charades-eval-VGG_GloVe-2024_05_01_00_25_50/charades_test_submission.jsonl"
        preds = load_result(pred_path)
        vid_dir = "../LGI4temporalgrounding/data/charades/raw_videos/"
        
    return preds, vid_dir

#----------------------------------------------------------------------------------------#

def compute_tiou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return float(intersection) / union

if __name__ == "__main__":
    dt = "charades" # among anet|charades
    preds, vid_dir = load_output(dt)

    make_result_video(preds, "Foo", dt, vid_dir, "Foo", visualize=False)



    