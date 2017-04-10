(ns dl4clj.nn.conf.layers.subsampling-layer
  (:import [org.deeplearning4j.nn.conf.layers SubsamplingLayer])
  (:require [dl4clj.nn.conf.layers.shared-fns :refer :all]))

(defn get-eps
  [& {:keys [this]}]
  (.getEps this))

(defn get-pnorm
  [& {:keys [this]}]
  (.getPnorm this))

;; all other methods are detailed in shared-fns ns
