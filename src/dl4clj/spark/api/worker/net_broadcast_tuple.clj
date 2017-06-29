(ns dl4clj.spark.api.worker.net-broadcast-tuple
  (:import [org.deeplearning4j.spark.api.worker NetBroadcastTuple])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn new-net-broadcast-tuple
  ;; this fn should not need to be used
  ;; gets called behind the scene
  ;; will be removed in the core branch
  "creation of a simple class for storing configurations,
  parameters and updaters in one class (so they can be broadcast together)

  :mln-conf (conf), a multi-layer configuration
   - see: dl4clj.nn.conf.builders.multi-layer-builders
     - requires a nn-conf with multiple layers
     - see: dl4clj.nn.conf.builders.nn-conf-builder

  :params (INDArray or vec), the parameters of the mln
   - see: (params) in dl4clj.nn.api.model

  :updater-state (INDArray or vec), the state of the updater attached to the mln-conf
   - see: (get-state-view-array) in dl4clj.nn.api.updater"
  [& {:keys [mln-conf params updater-state]}]
  (NetBroadcastTuple. mln-conf
                      (vec-or-matrix->indarray params)
                      (vec-or-matrix->indarray updater-state)))
