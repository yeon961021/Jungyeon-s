#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:15:48 2024

@author: ijeong-yeon
"""
from googleapiclient.discovery import build
from datetime import datetime
import pandas as pd
import re
from googleapiclient.errors import HttpError
import uuid

def data_collect(api_key, youtube, search_query, published_after, published_before):
    # Video Search
    search_response = youtube.search().list(
        q=search_query,
        part='id',
        type='video',
        publishedAfter=published_after,
        publishedBefore=published_before,
        maxResults=500,
        order='viewCount'
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]

    df_comments = pd.DataFrame(columns=['Video_ID', 'Comment_Number', 'Comment', 'Comment_ID', 'Author_Channel_ID'])
    english_pattern = re.compile(r'[a-zA-Z]')

    for video_id in video_ids:
        try:
            comment_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=500,
                order='relevance'
            ).execute()

            for item in comment_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                author_channel_id = comment['authorChannelId']['value']
                if english_pattern.search(comment['textDisplay']):
                    if int(comment['likeCount']) >= 0:
                        new_comment = {
                            'Video_ID': video_id,
                            'Comment_Number': item['id'],
                            'Comment': comment['textDisplay'],
                            'Comment_ID': author_channel_id,
                            'Author_Channel_ID': author_channel_id
                        }
                        df_comments = pd.concat([df_comments, pd.DataFrame([new_comment])], ignore_index=True)
        except HttpError as e:
            if e.resp.status == 403:
                print(f"Comment Blocked Video: {video_id}")
                continue

    df_comments['Video_Link'] = 'https://www.youtube.com/watch?v=' + df_comments['Video_ID']

    df_video_descriptions = pd.DataFrame(columns=['Video_ID', 'Video_Description', 'Video_Description_ID'])
    channel_ids = []

    for video_id in video_ids:
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()

        channel_id = video_response['items'][0]['snippet']['channelId']
        if channel_id not in channel_ids:
            channel_ids.append(channel_id)

        video_description = video_response['items'][0]['snippet']['description']
        new_row = {
            'Video_ID': video_id,
            'Video_Description': video_description,
            'Video_Description_ID': str(uuid.uuid4())
        }
        df_video_descriptions = pd.concat([df_video_descriptions, pd.DataFrame([new_row])], ignore_index=True)

    df_community_posts = pd.DataFrame(columns=['Channel_ID', 'Post_Title', 'Post_Content', 'Post_ID'])

    for channel_id in channel_ids:
        try:
            community_response = youtube.activities().list(
                part='snippet,contentDetails',
                channelId=channel_id,
                maxResults=1000,
                publishedAfter=published_after
            ).execute()
        except HttpError as e:
            continue

        for item in community_response.get('items', []):
            try:
                post_title = item['snippet']['title']
                post_content = item['snippet']['description']

                new_post = {
                    'Channel_ID': channel_id,
                    'Post_Title': post_title,
                    'Post_Content': post_content,
                    'Post_ID': str(uuid.uuid4())
                }
                df_community_posts = pd.concat([df_community_posts, pd.DataFrame([new_post])], ignore_index=True)

            except KeyError:
                pass

    return df_comments, df_video_descriptions, df_community_posts

"""
Data collection Pipeline
-> It extracts YouTube comments, video description and post title
"""
api_key = "API Key"  # Set your api key here
youtube = build('youtube', 'v3', developerKey=api_key) # Build a model for extacting data from YouTube

#"Tigray War" -> "Tigray" -> "TPLF" -> "Abiy Ahmed" -> "Ethiopian National Defense Force"
search_query = "Abiy Ahmed"

date_1_a = datetime(2020, 11, 3).strftime('%Y-%m-%dT%H:%M:%SZ') # Start Date
date_1_b = datetime(2020, 11, 9).strftime('%Y-%m-%dT%H:%M:%SZ') # End Date

a1, b1, c1 = data_collect(api_key, youtube, search_query, date_1_a, date_1_b)

comment = a1['Comment']
video_des = b1['Video_Description']
c_post = c1['Post_Title']
user_id = a1['Comment_ID']
combined_df = pd.concat([comment, user_id, video_des, c_post], axis=1)
print("Collected Data Points:", len(combined_df))
        
    
