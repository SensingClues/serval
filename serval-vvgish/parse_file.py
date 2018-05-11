# Copyright (C) 2017 DataArt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
from scipy.io import wavfile

print("**** Packages imported")

print("**** Add argument")
parser = argparse.ArgumentParser(description='Read file and process audio')
parser.add_argument('wav_file', type=str, help='File to read and process')

print("**** Function process file")
def process_file(wav_file):
    print("**** Read wav file")
    sr, data = wavfile.read(wav_file)
    print("**** Check data type")
    if data.dtype != np.int16:
        print("**** Data type is not int16")
        raise TypeError('Bad sample type: %r' % data.dtype)

    # local import to reduce start-up time
    print("**** Import audio processors")
    from audio.processor import WavProcessor, format_predictions

    print("**** Get predictions")
    with WavProcessor() as proc:
        predictions = proc.get_predictions(sr, data)

    print(format_predictions(predictions))

print("**** If statement")
if __name__ == '__main__':
    args = parser.parse_args()
    process_file(**vars(args))
