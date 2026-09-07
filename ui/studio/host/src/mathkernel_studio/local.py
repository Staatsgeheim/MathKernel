"""Studio presentation adapter for the independent local workflow owner."""
from .source import KernelSource, canonical
from .services import SessionScope
import hashlib
from mathkernel_workflow.contracts import descriptors

class LocalWorkflowSource(KernelSource):

    def __init__(self, kernel, runtime):
        from mathkernel import __version__
        super().__init__(kernel, host_id=runtime.host_id, workspace_id=runtime.workspace_id, version=__version__)
        self.runtime = runtime
        self.base_entries = self.entries
        self._refresh()

    def _refresh(self):
        self.entries = descriptors() + self.runtime.saved_descriptors() + self.base_entries
        self.catalog_revision = 'catalog_' + hashlib.sha256(canonical(self.entries)).hexdigest()[:24]

    def handshake(self):
        self._refresh()
        h = super().handshake()
        h['features']['result_inspect'] = True
        return h

    def _scope(self):
        return SessionScope(self.host_id, self.workspace_id, 'host-reader')

    def results(self, offset):
        return self.runtime.read('results', None, offset, self._scope())

    def result(self, ref):
        return self.runtime.read('results', ref, 0, self._scope())

    def result_admission(self, ref):
        return self.result(ref)
