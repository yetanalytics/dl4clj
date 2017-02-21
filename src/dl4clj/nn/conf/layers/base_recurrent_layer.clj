(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/BaseRecurrentLayer.html"}
  dl4clj.nn.conf.layers.base-recurrent-layer
  (:require [dl4clj.nn.conf.layers.feed-forward-layer :as ff-layer])
  (:import [org.deeplearning4j.nn.conf.layers BaseRecurrentLayer$Builder]))

(defn builder [^BaseRecurrentLayer$Builder builder opts]
  (ff-layer/builder builder opts))
