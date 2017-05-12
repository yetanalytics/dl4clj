(ns ^{:doc "implementation of the termination conditions found in the optimize package.

see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/terminations/package-summary.html"}
    dl4clj.optimize.termination.terminations
  (:import [org.deeplearning4j.optimize.terminations
            TerminationConditions
            Norm2Termination
            ZeroDirection
            EpsTermination])
  (:require [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method for creating instances of termination conditions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti termination-condition generic-dispatching-fn)

(defmethod termination-condition :eps [opts]
  (let [conf (:eps opts)
        {eps :eps
         tol :tolerance} conf]
    (if (contains-many? conf :eps :tolerance)
      (EpsTermination. eps tol)
      (EpsTermination.))))

(defmethod termination-condition :norm-2 [opts]
  (let [conf (:norm-2 opts)
        grad-tol (:gradient-tolerance conf)]
    (Norm2Termination. grad-tol)))

(defmethod termination-condition :zero-direction [opts]
  (ZeroDirection.))

(defmethod termination-condition :default [opts]
  (TerminationConditions.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns for creating termination conditions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-eps-termination-condition
  "Epsilon termination (absolute change based on tolerance)

  :eps (double), the value for epsilon

  :tolerance (double), the value for tolerance"
  [& {:keys [eps tolerance]
      :as opts}]
  (termination-condition {:eps opts}))

(defn new-norm-2-termination-condition
  "Terminate if the norm2 of the gradient is < a certain tolerance

  :gradient-tolerance (double), the tolerance"
  [& {:keys [gradient-tolerance]
      :as opts}]
  (termination-condition {:norm-2 opts}))

(defn new-zero-direction-termination-condition
  "terminates when the absolute magnitude of gradient is 0"
  []
  (termination-condition {:zero-direction {}}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api, also specified else where (termination-condition interface/api)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn time-to-terminate?
  "Whether to terminate based on the given meta data

  :termination-condition (termination-condition), created in this ns

  :cost (double), the current value of the cost fn

  :old-cost (double), the old value of the cost fn

  :other-params (java.lang.Object), other params...not sure what they are or should be"
  [terminate-condition cost old-cost other-params]
  (.terminate terminate-condition cost old-cost other-params))
