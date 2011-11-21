import unittest

from mantid.api import algorithm_mgr

from testhelpers import run_algorithm

class AlgorithmTest(unittest.TestCase):
  
    _load = None

    def setUp(self):
        if self._load is None:
            self.__class__._load = algorithm_mgr.create_unmanaged('Load')
            self._load.initialize()
  
    def test_alg_attrs_are_correct(self):
        self.assertTrue(self._load.name(), 'Load')
        self.assertTrue(self._load.version(), 1)
        self.assertTrue(self._load.category(), 'DataHandling')
    
        
    def test_alg_set_valid_prop_succeeds(self):
        self._load.set_property('Filename', 'LOQ48127.raw')
        
    def test_alg_set_invalid_prop_raises_error(self):
        alg = algorithm_mgr.create_unmanaged('Load')
        alg.initialize()
        args = ('Filename', 'nonexistent.txt')
        self.assertRaises(ValueError, alg.set_property, *args)
        
    def test_cannot_execute_with_invalid_properties(self):
        alg = algorithm_mgr.create_unmanaged('Load')
        alg.initialize()
        self.assertRaises(RuntimeError, alg.execute)
        
    def test_execute_succeeds_with_valid_props(self):
        data = [1.0,2.0,3.0]
        alg = run_algorithm('CreateWorkspace',DataX=data,DataY=data,NSpec=1,UnitX='Wavelength',child=True)
        self.assertEquals(alg.get_property('NSpec').value, 1)
        self.assertEquals(type(alg.get_property('NSpec').value), int)
        self.assertEquals(alg.get_property('NSpec').name, 'NSpec')
        ws = alg.get_property('OutputWorkspace').value
        self.assertTrue(ws.get_memory_size() > 0.0 )
