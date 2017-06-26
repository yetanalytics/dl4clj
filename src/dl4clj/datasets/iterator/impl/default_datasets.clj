(ns dl4clj.datasets.iterator.impl.default-datasets
  (:refer-clojure :exclude [reset!])
  (:import [org.deeplearning4j.datasets.iterator.impl
            CifarDataSetIterator IrisDataSetIterator LFWDataSetIterator
            MnistDataSetIterator RawMnistDataSetIterator]
           [java.util Random])
  (:require [dl4clj.utils :refer [contains-many?]]
            [datavec.api.split :refer :all]
            [nd4clj.linalg.api.ds-iter :refer :all]))
;; delete this ns
