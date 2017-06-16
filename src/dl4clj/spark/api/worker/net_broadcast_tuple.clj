(ns dl4clj.spark.api.worker.net-broadcast-tuple
  (:import [org.deeplearning4j.spark.api.worker NetBroadcastTuple]))

(defn new-net-broadcast-tuple
  ;; this fn should not need to be used
  ;; gets called behind the scene
  "creation of a simple class for storing configurations,
  parameters and updaters in one class (so they can be broadcast together)

  :mln-conf (conf), a multi-layer configuration
   - see: dl4clj.nn.conf.builders.multi-layer-builders
     - requires a nn-conf with multiple layers
     - see: dl4clj.nn.conf.builders.nn-conf-builder

  :params (INDArray), the parameters of the mln
   - see: (params) in dl4clj.nn.api.model

  :updater-state (INDArray), the state of the updater attached to the mln-conf
   - see: (get-state-view-array) in dl4clj.nn.api.updater"
  [& {:keys [mln-conf params updater-state]}]
  (NetBroadcastTuple. mln-conf params updater-state))
