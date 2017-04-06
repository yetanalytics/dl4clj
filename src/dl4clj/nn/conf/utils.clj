(ns dl4clj.nn.conf.utils
  (:import [org.deeplearning4j.nn.conf.layers InputTypeUtil]))

;;"https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/InputTypeUtil.html"
;;"https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/LayerValidation.html"
;;"https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/LayerBuilderTest.html"
;;"https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/LayerConfigValidationTest.html"
;;"https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/LayerConfigTest.html"

(defn contains-many? [m & ks]
  (every? #(contains? m %) ks))
