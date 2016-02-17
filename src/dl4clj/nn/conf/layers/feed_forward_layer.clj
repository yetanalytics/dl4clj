(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/FeedForwardLayer.html"}
  dl4clj.nn.conf.layers.feed-forward-layer
  (:require [dl4clj.nn.conf.layers.layer :as layer])
  (:import [org.deeplearning4j.nn.conf.layers FeedForwardLayer$Builder]))

(defn builder [^FeedForwardLayer$Builder builder {:keys [n-in nout
                                                         n-out nin]
                                                  :or {}
                                                  :as opts}]
  (layer/builder builder opts)
  (when (or n-in nin)
    (.nIn builder (or n-in nin)))
  (when (or n-out nout)
    (.nOut builder (or n-out nout)))
  builder)


