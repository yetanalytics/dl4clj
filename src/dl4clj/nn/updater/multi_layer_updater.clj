(ns ^{:doc "Gradient updater for MultiLayerNetworks.
 Expects backprop gradients for all layers to be in single Gradient object, keyed by "0_b", "1_w" etc.,
 as per MultiLayerNetwork.backward() -- NOT IMPLEMENTED YET

 see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/MultiLayerUpdater.html"}
    dl4clj.nn.updater.multi-layer-updater
  (:import [org.deeplearning4j.nn.updater MultiLayerUpdater]))

(defn new-multi-layer-updater
  "creates an instance of the MultiLayerUpdater class from dl4j.

  :mln (multi-layer-network), a multi layer network model
   - see: dl4clj.nn.conf.builders.multi-layer-builders

  :updater-state (INDArray), the state of the updater
   - need to test to figure out what should be in that array"
  [& {:keys [mln updater-state]
      :as opts}]
  (assert (contains? opts :mln)
          "you must provide a multi layer network")
  (if (contains? opts :updater-state)
    (MultiLayerUpdater. mln updater-state)
    (MultiLayerUpdater. mln)))

;; only uses methods from Updater interface
