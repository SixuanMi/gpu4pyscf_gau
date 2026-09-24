"""Compatibility with qzcli's local browser-login state and the training console API."""
import copy
import json
from pathlib import Path
import urllib.error
import urllib.request


def payload(platform, command):
    template = json.loads(Path(platform['resource_template']).expanduser().read_text())
    cfg = copy.deepcopy(template['framework_config'])
    price = cfg.pop('instance_spec_price_info')
    if price['gpu_count'] != 1:
        raise ValueError('This runner supports a single GPU per job')
    cfg['resource_spec_price'] = dict(cpu_type=price['cpu_info']['cpu_type'],
        cpu_count=price['cpu_count'], gpu_type=price['gpu_info']['gpu_type'], gpu_count=1,
        memory_size_gib=price['memory_size_gib'], quota_id=platform['spec'],
        logic_compute_group_id=platform['compute_group'])
    cfg.update(image=platform['image'], image_type=platform.get('image_type','SOURCE_PRIVATE'),
               instance_count=1, shm_gi=platform.get('shm_gib',64))
    return dict(name=platform['name'], workspace_id=platform['workspace'],
        project_id=platform['project'], logic_compute_group_id=platform['compute_group'],
        framework='pytorch', command=command, task_priority=platform.get('priority',6),
        auto_fault_tolerance=False, framework_config=[cfg])


def submit(platform, body):
    cookie_file = Path(platform.get('cookie_file','~/.qzcli/.cookie')).expanduser()
    try:
        cookie = json.loads(cookie_file.read_text())['cookie']
    except (OSError, KeyError):
        raise ValueError('qzcli login state unavailable; log in using qzcli on this host') from None
    base = platform.get('console_url','https://qz.sii.edu.cn').rstrip('/')
    if not base.startswith('https://'):
        raise ValueError('Qizhi console URL must use HTTPS')

    def post(path, data):
        req = urllib.request.Request(base+path, data=json.dumps(data).encode(),
            headers={'cookie':cookie,'content-type':'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in (401,403):
                raise ValueError('Qizhi login expired or insufficient permissions; refresh qzcli login') from None
            raise ValueError(f'Qizhi returned HTTP {exc.code}; inspect job history before retrying') from None
        except (OSError, ValueError):
            raise ValueError('Uncertain Qizhi response; inspect job history before retrying') from None

    history = post('/api/v1/train_job/list',dict(page_num=1,page_size=200,workspace_id=platform['workspace']))
    if history.get('code') != 0:
        raise ValueError('Cannot verify Qizhi job history; nothing submitted')
    data = history.get('data') or {}
    jobs = data.get('list') or data.get('jobs') or []
    existing = next((j for j in jobs if j.get('name') == platform['name']), None)
    if existing:
        raise ValueError(f'Job name already exists ({existing.get("job_id")}); choose a new name only for an intentional new run')
    response = post('/api/v2/train?Action=CreateJobConsole',body)
    job_id = (response.get('Result') or {}).get('job_id')
    if not job_id:
        raise ValueError('Submission not confirmed; inspect job history before retrying')
    return dict(job_id=job_id,name=platform['name'],workspace=platform['workspace'])
