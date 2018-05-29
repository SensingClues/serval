import hashlib
import json
import os
import sys


def md5sum(filename):
    """Computes the MD5 Hash for the contents of `filename`."""
    md5 = hashlib.md5()
    with open(filename, 'rb') as fin:
        for chunk in iter(lambda: fin.read(128 * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def dwnld(partition, mirror, shard):
 
    partition_parts = partition.split('/')

    assert mirror in {'us', 'eu', 'asia'}
    assert len(partition_parts) == 3
    assert partition_parts[1] in {'video_level', 'frame_level', 'video', 'frame'}
    assert partition_parts[2] in {'train', 'test', 'validate'}

    plan_url = 'https://storage.googleapis.com/data.yt8m.org/{}/download_plans/{}_{}.json'.format(partition_parts[0], partition_parts[1], partition_parts[2])

    num_shards = 1
    shard_id = 1
    
    shard_id, num_shards = shard.split(',')
    shard_id = int(shard_id)
    num_shards = int(num_shards)
    assert shard_id >= 1
    assert shard_id <= num_shards
    
    plan_filename = '%s_download_plan.json' % partition.replace('/', '_')
    os.system('curl %s > %s' % (plan_url, plan_filename))
    
    download_plan = json.loads(open(plan_filename).read())

    files = [f for f in download_plan['files'].keys()
           if int(hashlib.md5(f.encode('utf-8')).hexdigest(), 16) % num_shards == shard_id - 1]

    print ('Files remaining %i' % len(files))
    for f in files:
        print ('Downloading: %s' % f)
        if os.path.exists(f) and md5sum(f) == download_plan['files'][f]:
            print ('Skipping already downloaded file %s' % f)
            continue

        download_url = 'https://storage.googleapis.com/%s.data.yt8m.org/%s/%s' % (mirror, partition, f)
        os.system('curl %s > %s' % (download_url, f))
        if md5sum(f) == download_plan['files'][f]:
            print ('Successfully downloaded %s\n\n' % f)
            del download_plan['files'][f]
            open(plan_filename, 'w').write(json.dumps(download_plan))
        else:
            print ('Error downloading %s. MD5 does not match!\n\n' % f)

    print ('All done. No more files to download.')
    return
