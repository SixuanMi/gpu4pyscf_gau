"""Run a prepared case manifest with independent processes and NVML sampling."""
import ctypes as C
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class Memory(C.Structure):
    _fields_ = [('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]


def main():
    root = Path(sys.argv[1]).resolve()
    nv = C.CDLL('libnvidia-ml.so.1')
    assert nv.nvmlInit_v2() == 0
    handle = C.c_void_p()
    assert nv.nvmlDeviceGetHandleByIndex_v2(0,C.byref(handle)) == 0
    nv.nvmlDeviceGetMemoryInfo.argtypes = [C.c_void_p,C.POINTER(Memory)]
    cases = json.loads((root/'cases.json').read_text())
    results = []
    for name in cases:
        case = root/name
        (root/'progress.json').write_text(json.dumps(dict(active=name,finished=[r['case'] for r in results])))
        with (case/'process.log').open('w') as log, (case/'nvml.jsonl').open('w',buffering=1) as out:
            p = subprocess.Popen([sys.executable,str(Path(__file__).with_name('hessian_probe.py')),str(case)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start = time.monotonic()
            while p.poll() is None:
                if time.monotonic()-start > 1800:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                    break
                m = Memory()
                rc = nv.nvmlDeviceGetMemoryInfo(handle,C.byref(m))
                if rc == 0:
                    out.write(json.dumps(dict(time=time.time(),used=m.used,total=m.total))+'\n')
                time.sleep(.02)
            rc = p.wait()
        result = dict(case=name,returncode=rc)
        if (case/'result.json').exists():result.update(json.loads((case/'result.json').read_text()))
        else:result.update(completed=False,error='No result; inspect process.log')
        results.append(result)
        (root/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        # Stop after a failed small control; do not spend time on invalid large tests.
        if not result['completed'] and name.startswith(('water','oxygen')):
            break
    nv.nvmlShutdown()
    (root/'progress.json').write_text(json.dumps(dict(state='finished',count=len(results))))
    return 0 if len(results) == len(cases) and all(r['completed'] for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
