from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import os
import werkzeug
import glob
from flask import send_from_directory
from flask_restful import Resource, Api, reqparse



MEDIA_DIR = os.path.join(os.getcwd(), "media")
SEPARATION_DIR = os.path.join(os.getcwd(), "separation")
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


class AudioSeparation(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix",
                                       savedir=os.path.join('pretrained_models', 'sepformer-wsj03mix'))
        est_sources = model.separate_file(path=audiofile_path)

        torchaudio.save(os.path.join(SEPARATION_DIR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_DIR, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_DIR, "source3hat.wav"), est_sources[:, :, 2].detach().cpu(), 8000)
        separated_filenames = get_separated_filenames()
        return separated_filenames, 201


class AudioSeparationDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_DIR, filename, as_attachment=True)
