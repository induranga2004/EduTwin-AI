import { useEffect, useCallback } from 'react';
import { PANELS, ACTIONS } from '/imports/ui/components/layout/enums';
import KEYS from '/imports/utils/keys';
import Session from '/imports/ui/services/storage/in-memory';
import { DispatcherFunction } from '/imports/ui/components/layout/layoutTypes';

export const usePollShortcut = (layoutContextDispatch: DispatcherFunction, isPollingEnabled: boolean) => {
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    const {
      altKey, ctrlKey,
      metaKey, key,
    } = e;
    const isPollShortcut = altKey && key === KEYS.p && (ctrlKey || metaKey);

    if (isPollShortcut) {
      e.preventDefault();
      e.stopPropagation();

      if (Session.equals('pollInitiated', true)) {
        Session.setItem('resetPollPanel', true);
      }

      layoutContextDispatch({
        type: ACTIONS.SET_SIDEBAR_CONTENT_IS_OPEN,
        value: true,
      });
      layoutContextDispatch({
        type: ACTIONS.SET_SIDEBAR_CONTENT_PANEL,
        value: PANELS.POLL,
      });

      Session.setItem('forcePollOpen', true);
      Session.setItem('customPollShortcut', true);
    }
  }, [layoutContextDispatch]);

  useEffect(() => {
    if (isPollingEnabled) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
    return () => {};
  }, [isPollingEnabled, handleKeyDown]);
};

export default usePollShortcut;
