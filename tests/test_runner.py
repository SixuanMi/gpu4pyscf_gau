import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gpu4pyscf_gau.config import DEFAULT, load_config
from gpu4pyscf_gau.runner import Worker, gaussian_status, read_xyz, run_jobs
from gpu4pyscf_gau.cli import qz_command
from gpu4pyscf_gau.qzconsole import payload


FAKE_WORKER = r'''
import argparse,json,socket
from pathlib import Path
from gpu4pyscf_gau.protocol import read_input,format_output
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--config');p.add_argument('--socket');a=p.parse_args()
root=Path(a.root);count=0
with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
 server.bind(a.socket);server.listen();(root/'worker.ready').write_text('ready')
 while True:
  with server.accept()[0] as conn:
   req=json.loads(conn.makefile('rb').readline())
   if req['operation']=='reset':reply={'ok':True}
   else:
    count+=1;g=read_input(req['input']);n=len(g['numbers'])
    r=dict(energy_hartree=-1.,dipole_au=[0.,0.,0.],derivative_order=g['deriv'],
      gradient_hartree_bohr=[[0.,0.,0.]]*n,evaluation=count,scf_seconds=.1,gradient_seconds=0.,hessian_seconds=0.)
    Path(req['output']).write_text(format_output(r,g['deriv']))
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n'.replace('\\n','\n'))
    reply=dict(ok=True,result=r,summary={'evaluation':count})
   conn.sendall(json.dumps(reply).encode()+b'\n'.replace(b'\\n',b'\n'))
'''

FAKE_GAUSSIAN = r'''
import os,subprocess,sys
from pathlib import Path
text=sys.stdin.read()
if os.environ.get('LD_PRELOAD'):raise RuntimeError('CUDA preload leaked to Gaussian')
Path('external.in').write_text('3 0 0 1\n8 0 0 0 0\n1 0 1 0 0\n1 0 -1 0 0\n'.replace('\\n','\n'))
r=subprocess.run(['gpu_gau_external','R','external.in','external.out','external.msg'])
if r.returncode:sys.exit(r.returncode)
if os.environ.get('FAKE_GAUSSIAN_FAIL')=='1':
 print('Error termination');sys.exit(1)
Path('gaussian.chk').write_text('fake')
print('Normal termination of Gaussian')
'''


class RunnerTests(unittest.TestCase):
    def test_negligible_forces_and_failed_optimization(self):
        text=' Optimization completed on the basis of negligible forces.\nNormal termination of Gaussian'
        self.assertTrue(gaussian_status(text,'tsopt',0)['completed'])
        self.assertFalse(gaussian_status('Normal termination of Gaussian','opt',0)['completed'])
        self.assertFalse(gaussian_status(text,'tsopt',1)['completed'])

    def test_bad_config_and_multigeometry_xyz(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('{"gpu":{"with_hess":true}}')
            with self.assertRaises(ValueError):load_config(p)
            p.write_text('{"gpu":{"with_solvent":true}}')
            with self.assertRaises(ValueError):load_config(p)
            p.write_text('1\nx\nH 0 0 0\n1\nx\nH 1 0 0\n')
            with self.assertRaises(ValueError):read_xyz(p)

    def test_qz_quotes_paths_without_credential_access(self):
        p=dict(workspace='w',project='p',compute_group='g',spec='s',image='i',
               python='/some env/python',repository='/shared/my project',name='test')
        cmd=qz_command(p,['run','--config','/path with space/config.json'])
        import shlex
        inner=shlex.split(cmd[cmd.index('--command')+1])
        self.assertIn('/some env/python',inner)
        self.assertIn('/path with space/config.json',inner)
        self.assertNotIn('cookie',' '.join(cmd))

    def test_console_template_is_not_mutated_or_used_as_credentials(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'template.json'
            template=dict(framework_config=dict(cpu=20,gpu_count=1,mem_gi=200,
                instance_spec_price_info=dict(cpu_count=20,cpu_info={'cpu_type':''},
                gpu_count=1,gpu_info={'gpu_type':'test_gpu'},memory_size_gib=200)))
            file.write_text(json.dumps(template))
            p=dict(resource_template=str(file),name='test',workspace='w',project='p',
                   compute_group='g',spec='s',image='i')
            body=payload(p,'python run.py')
            self.assertEqual(body['framework_config'][0]['resource_spec_price']['quota_id'],'s')
            self.assertNotIn('cookie',json.dumps(body))
            self.assertEqual(json.loads(file.read_text()),template)

    def test_real_socket_callback_batch_archive_and_failure(self):
        with tempfile.TemporaryDirectory(prefix='bridge test ') as d:
            d=Path(d)
            worker=d/'fake_worker.py';worker.write_text(FAKE_WORKER)
            gauss=d/'fake g16';gauss.write_text('#!'+sys.executable+'\n'+FAKE_GAUSSIAN);gauss.chmod(0o700)
            xyz=d/'water.xyz';xyz.write_text('3\nwater\nO 0 0 0\nH 0 1 0\nH 0 -1 0\n')
            cfg=copy.deepcopy(DEFAULT);cfg['gaussian'].update(executable=str(gauss),formchk=None)
            cfg['runtime']['startup_timeout_seconds']=10
            cfg['runtime']['timeout_seconds']=10
            jobs=[dict(name=n,task='sp',xyz=str(xyz),charge=0,multiplicity=1) for n in ['one','two']]
            def command(self,python,directory):
                return [python,str(worker),'--root',str(directory),'--config',str(directory/'config.json'),'--socket',self.socket]
            with patch.object(Worker,'command',command):
                out=d/'results with spaces'
                ok=run_jobs(cfg,jobs,out)
                detail=(out/'summary.json').read_text()+'\n'+'\n'.join(p.read_text() for p in out.rglob('*.log'))
                self.assertTrue(ok,detail)
                summary=json.loads((out/'summary.json').read_text())['results']
                self.assertEqual([x['successful_evaluations'] for x in summary],[1,1])
                self.assertEqual(summary[1]['worker_startup_seconds'],0)
                self.assertTrue((out/'one/external.out').exists())
                self.assertFalse((out/'one/scratch').exists())
                with self.assertRaises(ValueError):run_jobs(cfg,jobs,out)
                cfg['gaussian']['environment']['FAKE_GAUSSIAN_FAIL']='1'
                self.assertFalse(run_jobs(cfg,jobs[:1],d/'failure'))
                self.assertFalse(json.loads((d/'failure/summary.json').read_text())['completed'])


if __name__=='__main__':unittest.main()
