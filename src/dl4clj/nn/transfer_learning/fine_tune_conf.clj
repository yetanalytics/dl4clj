(ns ^{:doc "a class for fine tuning a nn conf.
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/transferlearning/FineTuneConfiguration.html
and
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/transferlearning/FineTuneConfiguration.Builder.html"}
    dl4clj.nn.transfer-learning.fine-tune-conf
  (:import [org.deeplearning4j.nn.transferlearning FineTuneConfiguration$Builder
            FineTuneConfiguration])
  (:require [dl4clj.constants :as enum]
            [dl4clj.utils :refer [builder-fn eval-and-build]]
            [dl4clj.helpers :refer [value-of-helper]]
            [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; make the fine tuning conf
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def fine-tune-method-map
  {:activation-fn   '.activation
   :n-iterations    '.iterations
   :regularization? '.regularization
   :seed            '.seed})


(defn new-fine-tune-conf
  "creates a new fine tune configuration

  :activation-fn (keyword), the activation fn to change/add

  :n-iterations (int), the number of iterations to run

  :regularization? (boolean), should regularization be used?

  :seed (int or long), consistent randomization

  :eval-and-build? (boolean), do you want to evaluate and build the conf?
   - defaults to false"
  [& {:keys [activation-fn n-iterations regularization? seed as-code?]
      :or {as-code? true}
      :as opts}]
  (let [b `(FineTuneConfiguration$Builder.)
        a-fn (value-of-helper :activation-fn activation-fn)
        opts* (-> opts
                  (dissoc :activation-fn :as-code?)
                  (assoc :activation-fn a-fn))
        fn-chain (builder-fn b fine-tune-method-map opts*)]
    (if as-code?
      fn-chain
      (eval-and-build fn-chain))))
