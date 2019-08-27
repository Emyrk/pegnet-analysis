package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/FactomProject/factom"

	"github.com/pegnet/pegnet/common"
	"github.com/pegnet/pegnet/mining"
	"github.com/pegnet/pegnet/opr"
	log "github.com/sirupsen/logrus"
	"github.com/zpatrick/go-config"
)

type AltruisticMiner struct {
	config *config.Config

	// Factom blockchain related alerts
	FactomMonitor common.IMonitor
	OPRGrader     opr.IGrader

	EC *factom.ECAddress

	// Used when going over the network
	OPRMaker mining.IOPRMaker
}

func NewAltruisticMiner(config *config.Config, monitor common.IMonitor, grader opr.IGrader) *AltruisticMiner {
	c := new(AltruisticMiner)
	c.config = config
	c.FactomMonitor = monitor
	c.OPRGrader = grader
	k, err := config.Int("Miner.RecordsPerBlock")
	if err != nil {
		panic(err)
	}

	c.OPRMaker = mining.NewOPRMaker()

	err = c.PopulateECAddress()
	if err != nil {
		panic(err)
	}

	return c
}

func (w *AltruisticMiner) ECBalance() (int64, error) {
	return factom.GetECBalance(w.EC.String())
}

// PopulateECAddress only needs to be called once
func (w *AltruisticMiner) PopulateECAddress() error {
	// Get the Entry Credit Address that we need to write our OPR records.
	if ecadrStr, err := w.config.String("Miner.ECAddress"); err != nil {
		return err
	} else {
		ecAdr, err := factom.FetchECAddress(ecadrStr)
		if err != nil {
			return err
		}
		w.EC = ecAdr
	}
	return nil
}

func (c *AltruisticMiner) WriteEntry() {
	if w.oprTemplate == nil {
		return fmt.Errorf("no opr template")
	}

	operation := func() error {
		var err1, err2 error
		entry, err := w.oprTemplate.CreateOPREntry(unique.Nonce, unique.Difficulty)
		if err != nil {
			return err
		}

		_, err1 = factom.CommitEntry(entry, w.ec)
		_, err2 = factom.RevealEntry(entry)
		if err1 == nil && err2 == nil {
			return nil
		}

		return errors.New("Unable to commit entry to factom")
	}

	err := backoff.Retry(operation, common.PegExponentialBackOff())
	if err != nil {
		// TODO: Handle error in retry
		return err
	}
	return nil
}

func (c *AltruisticMiner) LaunchMiners(ctx context.Context) {
	opr.InitLX()
	mineLog := log.WithFields(log.Fields{"id": "altruism"})

	// TODO: Also tell Factom Monitor we are done listening
	alert := c.FactomMonitor.NewListener()
	gAlert := c.OPRGrader.GetAlert("coordinator")
	// Tell OPR grader we are no longer listening
	defer c.OPRGrader.StopAlert("coordinator")

	first := false
	mineLog.Info("Miners launched. Waiting for minute 1 to start mining...")
MiningLoop:
	for {
		var fds common.MonitorEvent
		select {
		case fds = <-alert:
		case <-ctx.Done(): // If cancelled
			return
		}

		hLog := mineLog.WithFields(log.Fields{
			"height": fds.Dbht,
			"minute": fds.Minute,
		})
		if !first {
			// On the first minute log how far away to mining
			hLog.Infof("On minute %d. %d minutes until minute 1 before mining starts.", fds.Minute, common.Abs(int(fds.Minute)-11)%10)
			first = true
		}

		hLog.Debug("Miner received alert")
		switch fds.Minute {
		case 1:
			// First check if we have the funds to mine
			bal, err := c.ECBalance()
			if err != nil {
				hLog.WithError(err).WithField("action", "balance-query").Error("failed to mine this block")
				continue MiningLoop // OPR cancelled
			}
			if bal == 0 {
				hLog.WithError(fmt.Errorf("entry credit balance is 0")).WithField("action", "balance-query").Error("will not mine, out of entry credits")
				continue MiningLoop // OPR cancelled
			}

			// Need to get an OPR record
			oprTemplate, err = c.OPRMaker.NewOPR(ctx, 0, fds.Dbht, c.config, gAlert)
			if err == context.Canceled {
				continue MiningLoop // OPR cancelled
			}
			if err != nil {
				hLog.WithError(err).Error("failed to mine this block")
				continue MiningLoop // OPR cancelled
			}

			// Get the OPRHash for miners to mine.
			oprHash = oprTemplate.GetHash()

			// The consolidator that will write to the blockchain
			c.FactomEntryWriter = c.FactomEntryWriter.NextBlockWriter()
			c.FactomEntryWriter.SetOPR(oprTemplate)

			// Submit our records! Yea, we aren't mining
			resps := c.FactomEntryWriter.AddMiner()

			buf := make([]byte, 8)
			binary.BigEndian.PutUint64(buf, oprTemplate.MinimumDifficulty)
			hLog.WithField("mindiff", fmt.Sprintf("%x", buf)).Info("Begin mining new OPR")

		}
	}
}
