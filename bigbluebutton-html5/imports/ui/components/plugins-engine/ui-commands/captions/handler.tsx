import { useEffect } from 'react';
import { CaptionsEnum, CaptionsLanguageEnum } from 'bigbluebutton-html-plugin-sdk/dist/cjs/ui-commands/captions/enums';
import { SetDisplayAudioCaptionsCommandArguments } from 'bigbluebutton-html-plugin-sdk/dist/cjs/ui-commands/captions/types';
import {
  setAudioCaptions,
  setUserLocaleProperty,
} from '/imports/ui/components/audio/audio-graphql/audio-captions/service';
import { useMutation } from '@apollo/client/react/hooks/useMutation';
import { SET_CAPTION_LOCALE } from '/imports/ui/core/graphql/mutations/userMutations';

const PluginCaptionsUiCommandsHandler = () => {
  const [setCaptionLocaleMutation] = useMutation(SET_CAPTION_LOCALE);
  const setUserCaptionLocale = (captionLocale: string, provider: string) => {
    setCaptionLocaleMutation({
      variables: {
        locale: captionLocale,
        provider,
      },
    });
  };

  const handleSetDisplayAudioCaptions = (event: CustomEvent<SetDisplayAudioCaptionsCommandArguments>) => {
    const { displayAudioCaptions } = event.detail;

    setUserLocaleProperty(
      displayAudioCaptions,
      setUserCaptionLocale,
    );
    setAudioCaptions(displayAudioCaptions !== CaptionsLanguageEnum.NONE);
  };

  useEffect(() => {
    window.addEventListener(
      CaptionsEnum.SET_DISPLAY_AUDIO_CAPTIONS,
      handleSetDisplayAudioCaptions as EventListener,
    );

    return () => {
      window.removeEventListener(
        CaptionsEnum.SET_DISPLAY_AUDIO_CAPTIONS,
        handleSetDisplayAudioCaptions as EventListener,
      );
    };
  }, []);
  return null;
};

export default PluginCaptionsUiCommandsHandler;
