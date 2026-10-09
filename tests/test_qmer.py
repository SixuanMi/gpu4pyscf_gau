import copy
import json
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import h5py
import numpy as np
from gpu4pyscf_gau.config import DEFAULT
from gpu4pyscf_gau.qmer import prepare, identity, irc_endpoints, stage, digest, run, bounded_results, cleanup_core_dumps


class QmerTests(unittest.TestCase):
    def dataset(self,p):
        with h5py.File(p,'w') as f:
            f.attrs.update(complete=True,charge=0,multiplicity=1,dataset_version='fixture')
            f.create_dataset('records/id',data=np.array([f'id{i}'.encode() for i in range(19)]))
            f.create_dataset('records/source_index',data=np.arange(19))
            f.create_dataset('records/id_num',data=np.array([f'test{i}'.encode() for i in range(19)]))
            f.create_dataset('offsets/atom',data=np.arange(20)*2)
            f.create_dataset('atoms/atomic_numbers',data=np.ones(38,dtype=int))
            a=f.create_dataset('TS/coordinates',data=np.zeros((38,3)));a.attrs['unit']='angstrom'

    def test_disjoint_complete_shards_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.dataset(r/'data.h5');m=prepare(r/'data.h5',r/'manifests',8,2)
            indices=[]
            for shard,name in enumerate(m['manifests']):
                rows=[json.loads(x) for x in (r/'manifests'/name).read_text().splitlines()]
                self.assertTrue(all(row['index']%8==shard and row['chunk']==i//2 for i,row in enumerate(rows)))
                indices.extend(row['index'] for row in rows)
            self.assertEqual(sorted(indices),list(range(19)))
            self.assertLessEqual(max(m['counts'])-min(m['counts']),1)
            with self.assertRaises(ValueError):prepare(r/'data.h5',r/'manifests')

    def test_recover_stage_and_changed_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);xyz=r/'input.xyz';xyz.write_text('1\nH\nH 0 0 0\n');out=r/'old';(out/'tsopt').mkdir(parents=True)
            (out/'summary.json').write_text('{"completed":true}')
            (out/'tsopt/final.xyz').write_text(xyz.read_text())
            cfg=copy.deepcopy(DEFAULT)
            fingerprint=digest(dict(config=cfg,task='tsopt',input=xyz.read_text(),charge=0,multiplicity=2))
            state=dict(stages={'tsopt':dict(status='running',fingerprint=fingerprint,output=str(out),attempts=[{}])})
            self.assertEqual(stage(cfg,r,'tsopt','tsopt',xyz,0,2,state,False,1),out)
            self.assertEqual(state['stages']['tsopt']['status'],'complete')
            cfg['gpu']['conv_tol']=1e-12
            with self.assertRaises(ValueError):stage(cfg,r,'tsopt','tsopt',xyz,0,2,state,False,2)

    def test_pipeline_and_complete_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.dataset(r/'data.h5');prepare(r/'data.h5',r/'manifests',8,2)
            (r/'config.json').write_text('{}')
            args=SimpleNamespace(config=str(r/'config.json'),output=str(r/'results'),
                manifests=str(r/'manifests'),shard=0,indices=[0],chunk=None,limit=1,
                imaginary_threshold=-20.,retry_failed=False,max_attempts=2)
            def fake_stage(cfg,reaction,name,task,xyz,charge,mult,state,retry,attempts):
                out=reaction/name;(out/task).mkdir(parents=True)
                (out/task/'final.xyz').write_text(Path(xyz).read_text())
                if task=='freq':
                    (out/'summary.json').write_text(json.dumps(dict(results=[dict(frequencies_cm1=([-100,100,200] if name=="ts_freq" else [100,200,300]))])))
                if task=='irc':
                    def scalar(label,n):return f'{label:<42} I {n}\n'
                    def array(label,v):return f'{label:<42} R N= {len(v)}\n'+' '.join(map(str,v))+'\n'
                    (out/task/'gaussian.fchk').write_text('title\nmethod\n'+scalar('IRC Num geometry variables',6)+scalar('IRC Num results per geometry',2)+array('IRC point       1 Results for each geome',[-1,0,-2,1,-3,-1])+array('IRC point       1 Geometries',list(range(18))))
                return out
            with patch('gpu4pyscf_gau.qmer.stage',side_effect=fake_stage) as mock:
                self.assertEqual(run(args),0)
                self.assertEqual([x.args[2] for x in mock.call_args_list],
                    ['tsopt','ts_freq','irc','endpoint_opt_reverse','endpoint_opt_forward','endpoint_freq_reverse','endpoint_freq_forward'])
                self.assertEqual(run(args),0)
                self.assertEqual(mock.call_count,7)
            state=json.loads(next((r/'results').rglob('state.json')).read_text())
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['ts_validation']['imaginary_count'],1)
            self.assertEqual(state['endpoint_validation']['reverse']['imaginary_count'],0)
            self.assertEqual(state['endpoint_validation']['forward']['imaginary_count'],0)
            args.indices=[1]
            with self.assertRaises(ValueError):run(args)

    def test_bounded_parallel_refill(self):
        barrier=threading.Barrier(4);refilled=threading.Event();lock=threading.Lock()
        active=0;peak=0
        def work(i):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active)
            try:
                if i<4:barrier.wait(timeout=5)
                if i==0:self.assertTrue(refilled.wait(timeout=5))
                if i==4:refilled.set()
                return i
            finally:
                with lock:active-=1
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(bounded_results(pool,range(5),lambda p,i:p.submit(work,i),4))
        self.assertEqual(sorted(x[1] for x in results),list(range(5)))
        self.assertEqual(peak,4)
        self.assertTrue(refilled.is_set())

    def test_core_cleanup_preserves_checkpoints_and_outside_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);inside=r/'reaction';inside.mkdir();(inside/'stage').mkdir()
            (inside/'stage/core.123').write_bytes(b'dump')
            (inside/'core').write_bytes(b'big dump')
            (inside/'core_model.json').write_text('keep')
            (inside/'gaussian.chk').write_bytes(b'checkpoint')
            outside=r/'core.456';outside.write_bytes(b'outside')
            (inside/'core.999').symlink_to(outside)
            report=cleanup_core_dumps(inside)
            self.assertEqual(report['bytes'],12)
            self.assertEqual(len(report['files']),2)
            self.assertTrue(outside.exists())
            self.assertTrue((inside/'core_model.json').exists())
            self.assertTrue((inside/'gaussian.chk').exists())

    def test_endpoint_imaginary_frequency_rejects_after_both_freqs(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.dataset(r/'data.h5');prepare(r/'data.h5',r/'manifests',8,2)
            (r/'config.json').write_text('{}')
            args=SimpleNamespace(config=str(r/'config.json'),output=str(r/'results'),
                manifests=str(r/'manifests'),shard=0,indices=[0],chunk=None,limit=1,
                imaginary_threshold=-20.,retry_failed=False,max_attempts=2)
            def fake_stage(cfg,reaction,name,task,xyz,*unused):
                out=reaction/name;(out/task).mkdir(parents=True)
                (out/task/'final.xyz').write_text(Path(xyz).read_text())
                if task=='freq':
                    values=[-100,100,200] if name in ['ts_freq','endpoint_freq_reverse'] else [100,200,300]
                    (out/'summary.json').write_text(json.dumps(dict(results=[dict(frequencies_cm1=values)])))
                return out
            def fake_endpoints(fchk,numbers,reaction):
                return {d:dict(xyz=str(reaction/'ts_input.xyz')) for d in ['reverse','forward']}
            with patch('gpu4pyscf_gau.qmer.stage',side_effect=fake_stage) as mock, patch('gpu4pyscf_gau.qmer.irc_endpoints',side_effect=fake_endpoints):
                self.assertEqual(run(args),1)
                self.assertEqual(mock.call_count,7)
            state=json.loads(next((r/'results').rglob('state.json')).read_text())
            self.assertEqual(state['status'],'rejected')
            self.assertEqual(state['endpoint_validation']['reverse']['imaginary_count'],1)
            self.assertEqual(state['endpoint_validation']['forward']['imaginary_count'],0)

    def test_signed_endpoints_from_accepted_fchk(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);p=r/'irc.fchk'
            def scalar(name,n):return f'{name:<42} I {n}\n'
            def array(name,values):return f'{name:<42} R N= {len(values)}\n'+' '.join(map(str,values))+'\n'
            p.write_text('title\nmethod\n'+scalar('IRC Num geometry variables',3)+scalar('IRC Num results per geometry',2)+array('IRC point       1 Results for each geome',[-1,0,-2,1,-3,2,-4,-1,-5,-2])+array('IRC point       1 Geometries',list(range(15))))
            d=irc_endpoints(p,[1],r)
            self.assertEqual(d['forward']['reaction_coordinate'],2)
            self.assertEqual(d['reverse']['reaction_coordinate'],-2)
            self.assertIn('3.175063265418',(r/'forward.xyz').read_text())
            p.write_text(p.read_text().replace('-4 -1 -5 -2','-4 1 -5 2'))
            with self.assertRaises(ValueError):irc_endpoints(p,[1],r)


if __name__=='__main__':unittest.main()
