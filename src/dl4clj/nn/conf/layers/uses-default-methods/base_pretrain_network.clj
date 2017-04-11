(ns dl4clj.nn.conf.layers.uses-default-methods.base-pretrain-network
  (:import [org.deeplearning4j.nn.conf.layers BasePretrainNetwork])
  (:require [dl4clj.nn.conf.layers.shared-fns :refer :all]
            [dl4clj.nn.conf.builders.builders :as b]))



;; this ns does not add any new layer manipulation fns.
;; see layers.shared-fns for these fns
