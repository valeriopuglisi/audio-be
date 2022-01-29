from flask import Flask, send_file, send_from_directory
from flask_restful import Resource, Api, reqparse
import os
import werkzeug
import glob

MEDIA_DIR = os.path.join(os.getcwd(), "media")
# print(MEDIA_DIR)
# print(SEPARATION_DIR)
media_files = glob.glob(os.path.join(os.getcwd(), "media", "*"))
# print(media_files)
# All files and directories ending with .txt and that don't begin with a dot:
AudioFiles = {}
for file in media_files:
    title = file.split("\\")[-1]
    if len(AudioFiles.keys()):
        audio_id = int(max(AudioFiles.keys()).lstrip('audio')) + 1
    else:
        audio_id = 0

    audio_id = 'audio%i' % audio_id
    AudioFiles[audio_id] = {'title': title}


def get_separated_filenames():
    separated_filename_paths = glob.glob(os.path.join(os.getcwd(), "separation", "*"))
    separated_filenames = []
    for file in separated_filename_paths:
        separated_filenames.append(file.split("\\")[-1])
    return separated_filenames


class AudioFilesList(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        return AudioFiles

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile.save(os.path.join(MEDIA_DIR, audiofile.filename))
        print(AudioFiles)
        audio_keys = AudioFiles.keys()
        print(audio_keys)
        audio_id = int(max(audio_keys).lstrip('audio')) + 1
        audio_id = 'audio%i' % audio_id
        AudioFiles[audio_id] = {'title': args['title']}
        return AudioFiles, 201


class AudioFileDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(MEDIA_DIR, filename)
        print(filename_path)
        return send_from_directory(MEDIA_DIR, filename, as_attachment=True)

