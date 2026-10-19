import { useEffect, useRef } from 'react';
import logger from '/imports/startup/client/logger';

export const useJoinLogger = (meetingId: string, meetingName: string, isBreakout: boolean) => {
  const isJoinLoggedRef = useRef<boolean>(false);

  useEffect(() => {
    if (!isJoinLoggedRef.current && meetingId) {
      const logMessage = isBreakout ? 'User joined breakout room' : 'User joined main room';
      logger.info({
        logCode: 'app_component_componentdidmount',
        extraInfo: { meetingId, meetingName },
      }, logMessage);
      isJoinLoggedRef.current = true;
    }
  }, [meetingId, meetingName, isBreakout]);
};

export default useJoinLogger;
