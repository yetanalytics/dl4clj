(ns ^{:doc "Interface for early stopping trainers.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/IEarlyStoppingTrainer.html"}
    dl4clj.earlystopping.interfaces.early-stopping-trainer
  (:import [org.deeplearning4j.earlystopping.trainer IEarlyStoppingTrainer]))

(defn fit-trainer!
  "Conduct early stopping training

  returns an early stopping result"
  [trainer]
  (.fit trainer))

(defn set-trainer-listeners!
  "Set the early stopping listener

  returns the trainer

  :listener (listener), a listener object that implements the
  early-stopping-listener interface
  - see : TBD"
  [& {:keys [trainer listener]}]
  (doto trainer (.setListener listener)))
