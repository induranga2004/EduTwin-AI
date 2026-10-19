import React from 'react';
import KEYS from '/imports/utils/keys';
import Styles from './styles';

interface ExternalVideoOverlayProps {
  onVerticalArrow?: () => void;
}

const ExternalVideoOverlay: React.FC<ExternalVideoOverlayProps> = (props) => {
  const { onVerticalArrow } = props;

  return (
    <Styles.ExternalVideoOverlay
      id="external-video-overlay"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.target !== e.currentTarget) return;

        if ([KEYS.ARROW_DOWN, KEYS.ARROW_UP].includes(e.key)) {
          onVerticalArrow?.();
        }
      }}
    />
  );
};

export default ExternalVideoOverlay;
